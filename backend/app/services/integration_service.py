"""Integration lifecycle and the ``sync_integration`` job."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import OrgContext
from app.config import get_settings
from app.connectors import integrations as ic
from app.connectors import oauth
from app.core.security import constant_time_equals, encrypt_secret
from app.db.session import tenant_session
from app.jobs.queue import JobContext, PermanentJobError, enqueue, handler
from app.llm.registry import credential_value
from app.models import Credential, Document, DocumentVersion, Integration, KnowledgeBase, Organization, SyncRun, User
from app.rbac.authorizer import Authorizer, Principal
from app.services.activity import audit, emit
from app.storage import get_storage, make_key
from app.utils.errors import CSVAIError, NotFound, ValidationFailed


def _now() -> datetime:
    return datetime.now(UTC)


def get(ctx: OrgContext, integration_id: uuid.UUID) -> Integration:
    i = ctx.db.get(Integration, integration_id)
    if i is None or i.org_id != ctx.org_id:
        raise NotFound("Integration not found.")
    return i


def store_secret(db: Session, org_id: uuid.UUID, user_id: uuid.UUID | None, i: Integration, secret: str) -> None:
    """Encrypt ``secret`` into the integration's Credential (created on first use)."""
    if i.credential_id:
        cred = db.get(Credential, i.credential_id)
        if cred:
            cred.ciphertext, cred.rotated_at = encrypt_secret(secret), _now()
            return
    cred = Credential(org_id=org_id, kind=f"integration:{i.provider}", ciphertext=encrypt_secret(secret), created_by=user_id)
    db.add(cred)
    db.flush()
    i.credential_id = cred.id


def set_secret(ctx: OrgContext, i: Integration, secret: str | None) -> None:
    if not secret:
        return
    store_secret(ctx.db, ctx.org_id, ctx.user_id, i, secret)


def drop_secret(db: Session, i: Integration) -> None:
    if i.credential_id:
        cred = db.get(Credential, i.credential_id)
        i.credential_id = None
        if cred:
            db.delete(cred)


# ------------------------------------------------------------------------- auth modes


def auth_modes(provider: str) -> list[str]:
    return list(ic.INTEGRATIONS.get(provider, {}).get("auth_modes") or ["token"])


def normalise_config(provider: str, config: dict[str, Any] | None, secret: str | None) -> dict[str, Any]:
    """Pin ``auth_mode`` for providers that authenticate in more than one way."""
    config = dict(config or {})
    modes = auth_modes(provider)
    if "oauth" in modes and len(modes) > 1 and not config.get("auth_mode"):
        config["auth_mode"] = "token" if secret else "oauth"
    elif modes == ["oauth"]:
        config["auth_mode"] = "oauth"
    return config


def oauth_key(provider: str, config: dict[str, Any] | None) -> str | None:
    """The OAuth vendor key when this integration authenticates with OAuth, else None."""
    meta = ic.INTEGRATIONS.get(provider, {})
    modes = meta.get("auth_modes") or []
    if "oauth" not in modes:
        return None
    return meta["oauth"]["provider"] if ((config or {}).get("auth_mode") or modes[0]) == "oauth" else None


def oauth_state(i: Integration) -> dict[str, Any] | None:
    """Non-secret OAuth status for API responses (tokens never leave the Credential)."""
    key = oauth_key(i.provider, i.config)
    if key is None:
        return None
    info = (i.cursor or {}).get("oauth") or {}
    return {"provider": key, "vendor": oauth.PROVIDERS[key].vendor, "connected": bool(i.credential_id) and bool(info), "account": info.get("account"), "connected_at": info.get("connected_at")}


def catalog() -> list[dict[str, Any]]:
    out = []
    for key, meta in ic.INTEGRATIONS.items():
        entry: dict[str, Any] = {"key": key, **meta}
        o = meta.get("oauth")
        if o:
            p = oauth.PROVIDERS[o["provider"]]
            missing = oauth.missing_settings(p.key)
            entry["oauth"] = {**o, "vendor": p.vendor, "configured": not missing, "redirect_uri": oauth.redirect_uri(), "env": oauth.env_names(p) + (["OAUTH_MICROSOFT_TENANT (optional, default common)"] if p.key == "microsoft" else []), "console": p.console}
            entry["needs_setup"] = missing
        out.append(entry)
    return out


def require_oauth_app(provider: str) -> None:
    key = (ic.INTEGRATIONS.get(provider, {}).get("oauth") or {}).get("provider")
    if key:
        oauth.require_app(key)  # raises NotAvailable naming the missing env vars


# ------------------------------------------------------------------------------ OAuth


def oauth_start(ctx: OrgContext, i: Integration) -> str:
    key = oauth_key(i.provider, i.config)
    if key is None:
        raise ValidationFailed("This integration authenticates with a token, not OAuth.")
    require_oauth_app(i.provider)
    ic.build(i.provider, i.config, None).validate()
    url, nonce = oauth.authorization_url(provider_key=key, scopes=ic.INTEGRATIONS[i.provider]["oauth"]["scopes"], org_id=ctx.org_id, user_id=ctx.user_id, integration_id=i.id, integration_provider=i.provider, config=i.config)
    # Single use without a migration: the callback must present this nonce, and clears it.
    i.cursor = {**(i.cursor or {}), "oauth_pending": {"nonce": oauth.nonce_hash(nonce), "by": str(ctx.user_id), "at": _now().isoformat()}}
    return url


def _back(**params: str | None) -> str:
    return get_settings().public_app_url.rstrip("/") + "/integrations?" + urlencode({k: v for k, v in params.items() if v})


def oauth_callback(*, code: str | None, state: str | None, error: str | None, error_description: str | None) -> str:
    """Finish the authorization-code flow; returns the frontend URL to redirect to.

    Authenticated by the signed state alone (a browser redirect carries no bearer token):
    signature + expiry, the single-use nonce, and the user's *current* ``org.integrations``.
    """
    if not state:
        return _back(oauth="error", message="The authorization response had no state. Start again from Integrations.")
    try:
        claims = oauth.decode_state(state)
        org_id, user_id, iid = uuid.UUID(claims["org"]), uuid.UUID(claims["sub"]), uuid.UUID(claims["iid"])
    except jwt.ExpiredSignatureError:
        return _back(oauth="error", message=f"The authorization link expired after {oauth.STATE_MINUTES} minutes. Start again.")
    except (jwt.InvalidTokenError, ValueError):
        return _back(oauth="error", message="The authorization state is invalid. Start again from Integrations.")
    with tenant_session(org_id, user_id) as db:
        i = db.get(Integration, iid)
        if i is None or i.org_id != org_id or i.provider != claims["prv"]:
            return _back(oauth="error", message="The integration no longer exists.")
        pending = (i.cursor or {}).get("oauth_pending") or {}
        if not pending.get("nonce") or not constant_time_equals(str(pending["nonce"]), oauth.nonce_hash(str(claims["nonce"]))):
            audit(db, "integration.oauth_connect", org_id=org_id, actor_user_id=user_id, resource_type="integration", resource_id=iid, outcome="denied", meta={"reason": "state already used or superseded"})
            return _back(oauth="error", integration=str(iid), message="This authorization was already used or replaced by a newer one. Start again.")
        i.cursor = {k: v for k, v in (i.cursor or {}).items() if k != "oauth_pending"}
        db.commit()  # consume the state before talking to the vendor
        user = db.get(User, user_id)
        org = db.get(Organization, org_id)
        authz = Authorizer(db, Principal(user_id=user_id, email=user.email if user else "", org_id=org_id)).load()
        if user is None or user.status != "active" or org is None or org.status != "active" or not authz.is_member or not authz.has("org.integrations"):
            audit(db, "integration.oauth_connect", org_id=org_id, actor_user_id=user_id, resource_type="integration", resource_id=iid, outcome="denied", meta={"reason": "missing org.integrations"})
            return _back(oauth="error", integration=str(iid), message="You no longer have permission to manage integrations in this organization.")
        vendor = oauth.PROVIDERS[claims["op"]].vendor
        actor: dict[str, Any] = {"org_id": org_id, "actor_user_id": user_id, "actor_email": user.email, "resource_type": "integration", "resource_id": iid}
        if error or not code:
            message = f"Authorization was cancelled at {vendor}." if error == "access_denied" else f"{vendor} returned an error: {(error_description or error or 'no authorization code')[:200]}"
            audit(db, "integration.oauth_connect", **actor, outcome="failure", meta={"error": (error or "no_code")[:80]})
            return _back(oauth="error", integration=str(iid), message=message)
        try:
            tokens = oauth.exchange_code(provider_key=claims["op"], code=code, verifier=oauth.state_verifier(claims), scopes=ic.INTEGRATIONS[i.provider]["oauth"]["scopes"], tenant=claims.get("tnt"))
            tokens = oauth.complete(tokens, i.config)
        except CSVAIError as exc:
            i.status, i.last_error = "failed", exc.message[:2000]
            audit(db, "integration.oauth_connect", **actor, outcome="failure", meta={"reason": exc.message[:200]})
            emit(db, org_id, "connection.failed", resource_type="integration", resource_id=iid, actor_user_id=user_id, severity="error", message=f"{i.name}: {exc.message[:200]}")
            return _back(oauth="error", integration=str(iid), message=exc.message[:300])
        store_secret(db, org_id, user_id, i, oauth.dump(tokens))
        info = {"provider": claims["op"], "account": oauth.account_label(tokens), "connected_at": _now().isoformat(), "connected_by": str(user_id)}
        i.cursor = {**(i.cursor or {}), "oauth": info}
        audit(db, "integration.oauth_connect", **actor, meta={"provider": i.provider, "scope": str(tokens.get("scope") or "")[:500]})

        def _save(secret: str) -> None:
            store_secret(db, org_id, user_id, i, secret)

        connector = ic.build(i.provider, i.config, oauth.dump(tokens), save_secret=_save)
        try:
            result = connector.test()
        except CSVAIError as exc:
            i.status, i.last_error = "failed", exc.message[:2000]
            emit(db, org_id, "connection.failed", resource_type="integration", resource_id=iid, actor_user_id=user_id, severity="error", message=f"{i.name}: {exc.message[:200]}")
            return _back(oauth="error", integration=str(iid), message=f"Authorized with {vendor}, but the test failed: {exc.message}"[:300])
        finally:
            connector.close()
        if result.get("account"):
            i.cursor = {**i.cursor, "oauth": {**info, "account": str(result["account"])[:200]}}
        i.status, i.last_error, i.last_success_at = "connected", None, _now()
        schedule(i)
        emit(db, org_id, "connection.tested", resource_type="integration", resource_id=iid, actor_user_id=user_id, severity="success", message=f"{i.name} connected to {vendor}")
    return _back(oauth="connected", integration=str(iid))


def schedule(i: Integration) -> None:
    authorized = i.credential_id is not None or not oauth_key(i.provider, i.config)  # OAuth: nothing to run until connected
    i.next_sync_at = _now() + timedelta(minutes=i.sync_interval_minutes) if i.sync_interval_minutes and i.enabled and not i.sync_paused and authorized else None


def test(ctx: OrgContext, i: Integration) -> dict[str, Any]:
    key = oauth_key(i.provider, i.config)
    if key and not i.credential_id:
        return {"ok": False, "error": f"Not authorized yet. Use “Connect with {oauth.PROVIDERS[key].vendor}” first.", "code": "not_authorized"}
    connector = ic.build(i.provider, i.config, credential_value(ctx.db, i.credential_id), save_secret=lambda s: set_secret(ctx, i, s))
    previous = i.status
    try:
        result = connector.test()
    except CSVAIError as exc:
        i.status, i.last_error = "failed", exc.message
        emit(ctx.db, ctx.org_id, "connection.failed", resource_type="integration", resource_id=i.id, actor_user_id=ctx.user_id, severity="error", message=f"{i.name}: {exc.message}")
        return {"ok": False, "error": exc.message, "code": exc.code}
    finally:
        connector.close()
    i.status, i.last_error = "connected", None
    i.last_success_at = _now()
    emit(ctx.db, ctx.org_id, "connection.restored" if previous == "failed" else "connection.tested", resource_type="integration", resource_id=i.id, actor_user_id=ctx.user_id, severity="success", message=f"{i.name} connected")
    return result


def sync_now(ctx: OrgContext, i: Integration, mode: str = "incremental") -> uuid.UUID:
    job = enqueue(ctx.db, ctx.org_id, "sync_integration", payload={"source_id": str(i.id), "mode": mode}, created_by=ctx.user_id, resource_type="integration", resource_id=i.id, idempotency_key=f"isync:{i.id}:{int(time.time() * 1000)}")
    return job.id


@handler("sync_integration")
def sync_integration(ctx: JobContext) -> dict[str, Any]:
    iid = uuid.UUID(ctx.payload["source_id"])
    mode = ctx.payload.get("mode", "incremental")
    with ctx.session() as db:
        i = db.get(Integration, iid)
        if i is None or not i.enabled:
            return {"skipped": "integration missing or disabled"}
        if i.kb_id is None or db.get(KnowledgeBase, i.kb_id) is None:
            raise CSVAIError("Choose a target knowledge base before synchronising.")
        provider, config, name, kb_id = i.provider, dict(i.config), i.name, i.kb_id
        secret = credential_value(db, i.credential_id)
        if oauth_key(provider, config) and not secret:
            raise PermanentJobError(f"{name} is not authorized yet: connect it from Integrations first.")
        known = {d.source_ref: (d.id, (d.meta or {}).get("external_version")) for d in db.scalars(select(Document).where(Document.source_type == "integration", Document.source_id == iid))}
        run = SyncRun(org_id=ctx.org_id, source_type="integration", source_id=iid, job_id=ctx.job_id, mode=mode)
        db.add(run)
        i.status = "syncing"
        db.flush()
        run_id = run.id
        emit(db, ctx.org_id, "sync.started", resource_type="integration", resource_id=iid, actor_user_id=ctx.created_by, message=f"Sync started: {name}", payload={"run_id": str(run_id)})
    stats = {"seen": 0, "added": 0, "updated": 0, "deleted": 0, "failed": 0, "unchanged": 0}
    seen: set[str] = set()

    def _persist_secret(new_secret: str) -> None:
        """Rotated OAuth tokens, saved in their own transaction mid-sync."""
        with ctx.session() as sdb:
            row = sdb.get(Integration, iid)
            if row is not None:
                store_secret(sdb, ctx.org_id, ctx.created_by, row, new_secret)

    connector: ic.Connector | None = None
    try:
        connector = ic.build(provider, config, secret, save_secret=_persist_secret)
        for item in connector.items():
            stats["seen"] += 1
            seen.add(item.external_id)
            prev = known.get(item.external_id)
            if prev and prev[1] == item.version and mode != "full":
                stats["unchanged"] += 1
                continue
            try:
                fetched = connector.fetch(item)
            except CSVAIError:
                stats["failed"] += 1
                continue
            fd, path = tempfile.mkstemp()
            os.close(fd)
            try:
                with open(path, "wb") as fh:
                    fh.write(fetched.content)
                key = make_key(ctx.org_id, "documents", fetched.filename)
                size = get_storage().put_file(key, path)
            finally:
                os.unlink(path)
            sha = hashlib.sha256(fetched.content).hexdigest()
            with ctx.session() as db:
                doc = db.get(Document, prev[0]) if prev else None
                from app.ingestion.detect import MIME_BY_EXT, extension_of

                ext = extension_of(fetched.filename)
                if doc is None:
                    doc = Document(org_id=ctx.org_id, kb_id=kb_id, title=item.title[:500], filename=fetched.filename[:500], ext=ext, mime_type=MIME_BY_EXT.get(ext, "text/html"), owner_id=ctx.created_by, source_type="integration", source_id=iid, source_ref=item.external_id, source_url=item.url, size_bytes=size, sha256=sha, storage_key=key)
                    db.add(doc)
                    db.flush()
                    stats["added"] += 1
                else:
                    doc.current_version += 1
                    doc.storage_key, doc.size_bytes, doc.sha256 = key, size, sha
                    stats["updated"] += 1
                doc.status = "queued"
                doc.meta = {**(doc.meta or {}), "external_version": item.version, "integration": name}
                db.add(DocumentVersion(org_id=ctx.org_id, document_id=doc.id, version=doc.current_version, sha256=sha, size_bytes=size, storage_key=key, filename=doc.filename, created_by=ctx.created_by, note="integration sync"))
                enqueue(db, ctx.org_id, "ingest_document", payload={"document_id": str(doc.id)}, created_by=ctx.created_by, resource_type="document", resource_id=doc.id, idempotency_key=f"ingest:{doc.id}:v{doc.current_version}")
            if stats["seen"] % 10 == 0:
                ctx.progress(min(0.95, stats["seen"] / max(int(config.get("max_pages", config.get("max_items", 100))), 1)), f"{stats['seen']} items")
        # Deletion propagation: items that disappeared from the source.
        removed = [ref for ref in known if ref not in seen]
        if removed:
            from app.services.knowledge_service import _delete_objects, _scrub_citations, _storage_keys_for

            with ctx.session() as db:
                for ref in removed:
                    doc = db.get(Document, known[ref][0])
                    if doc is None:
                        continue
                    keys = _storage_keys_for(db, document_id=doc.id)
                    _scrub_citations(db, [doc.id])
                    db.delete(doc)
                    db.flush()
                    _delete_objects(keys)
                    stats["deleted"] += 1
    except Exception as exc:
        if connector is not None:
            connector.close()
        message = getattr(exc, "message", None) or str(exc)
        with ctx.session() as db:
            run = db.get(SyncRun, run_id)
            i = db.get(Integration, iid)
            run.status, run.finished_at, run.error = "failed", _now(), message[:2000]
            run.items_seen, run.items_added, run.items_updated, run.items_deleted, run.items_failed = stats["seen"], stats["added"], stats["updated"], stats["deleted"], stats["failed"]
            if i:
                i.status, i.last_error = "failed", message[:2000]
                schedule(i)
            emit(db, ctx.org_id, "sync.failed", resource_type="integration", resource_id=iid, severity="error", message=f"Sync failed: {name}: {message[:200]}", payload={"run_id": str(run_id)})
        raise
    connector.close()
    with ctx.session() as db:
        run = db.get(SyncRun, run_id)
        i = db.get(Integration, iid)
        run.status, run.finished_at = "succeeded", _now()
        run.items_seen, run.items_added, run.items_updated, run.items_deleted, run.items_failed = stats["seen"], stats["added"], stats["updated"], stats["deleted"], stats["failed"]
        i.status, i.last_error = "connected", None
        i.last_sync_at = i.last_success_at = _now()
        schedule(i)
        emit(db, ctx.org_id, "sync.completed", resource_type="integration", resource_id=iid, severity="success", message=f"Sync completed: {name} (+{stats['added']} ~{stats['updated']} −{stats['deleted']})", payload={"run_id": str(run_id), **stats})
    return stats
