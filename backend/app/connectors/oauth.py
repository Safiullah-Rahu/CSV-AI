"""OAuth 2.0 authorization-code flow (PKCE S256 + signed state) for SaaS integrations.

* OAuth apps are platform-level (``OAUTH_<VENDOR>_CLIENT_ID`` / ``_SECRET``); an
  unconfigured vendor is reported as ``needs_setup`` and cannot be started.
* ``state`` is a signed JWT (10 min) carrying org, user, integration, vendor, a
  single-use nonce (its hash is kept in ``integration.cursor`` until the
  callback) and the PKCE verifier **encrypted** with the platform key.
* Tokens are stored as encrypted JSON in the integration's ``Credential``:
  ``{"type": "oauth", "provider", "access_token", "refresh_token", "expires_at",
  "scope", "extras": {...}}``. They are never returned by the API or logged.
* Expired access tokens are refreshed with the refresh-token grant; rotated
  tokens are persisted through the connector's ``save_secret`` callback.

Every OAuth server must ignore unknown parameters (RFC 6749 §3.1), so PKCE is
sent to all vendors; the ones that support it (Google, Microsoft, Dropbox,
Atlassian, Slack) enforce it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse

import jwt

from app.config import get_settings
from app.connectors.vendor_http import VendorHTTP, vendor_error
from app.core.security import create_purpose_token, decode_token, decrypt_secret, encrypt_secret
from app.utils.errors import ConnectorError, NotAvailable, ValidationFailed

STATE_TYPE = "oauth_state"
STATE_MINUTES = 10
CALLBACK_PATH = "/api/v1/integrations/oauth/callback"


@dataclass(frozen=True)
class OAuthProvider:
    key: str
    vendor: str
    auth_url: str  # may contain {tenant}
    token_url: str
    env: str  # settings/env prefix, e.g. "oauth_google" → OAUTH_GOOGLE_CLIENT_ID
    auth_params: dict[str, str] = field(default_factory=dict)
    scope_sep: str = " "
    token_auth: str = "body"  # body | basic
    token_format: str = "form"  # form | json
    console: str = ""  # where an operator registers the app


PROVIDERS: dict[str, OAuthProvider] = {
    "google": OAuthProvider("google", "Google", "https://accounts.google.com/o/oauth2/v2/auth", "https://oauth2.googleapis.com/token", "oauth_google", {"access_type": "offline", "prompt": "consent", "include_granted_scopes": "true"}, console="Google Cloud console → APIs & Services → Credentials (OAuth client, Web application)"),
    "microsoft": OAuthProvider("microsoft", "Microsoft", "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize", "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token", "oauth_microsoft", {"response_mode": "query", "prompt": "select_account"}, console="Microsoft Entra admin center → App registrations (Web platform)"),
    "dropbox": OAuthProvider("dropbox", "Dropbox", "https://www.dropbox.com/oauth2/authorize", "https://api.dropboxapi.com/oauth2/token", "oauth_dropbox", {"token_access_type": "offline"}, console="Dropbox App Console (scoped access)"),
    "notion": OAuthProvider("notion", "Notion", "https://api.notion.com/v1/oauth/authorize", "https://api.notion.com/v1/oauth/token", "oauth_notion", {"owner": "user"}, token_auth="basic", token_format="json", console="Notion → My integrations → Public integration"),
    "atlassian": OAuthProvider("atlassian", "Atlassian", "https://auth.atlassian.com/authorize", "https://auth.atlassian.com/oauth/token", "oauth_atlassian", {"audience": "api.atlassian.com", "prompt": "consent"}, token_format="json", console="Atlassian developer console → OAuth 2.0 (3LO) integration"),
    "slack": OAuthProvider("slack", "Slack", "https://slack.com/oauth/v2/authorize", "https://slack.com/api/oauth.v2.access", "oauth_slack", scope_sep=",", console="api.slack.com/apps → OAuth & Permissions"),
}

ATLASSIAN_RESOURCES = "https://api.atlassian.com/oauth/token/accessible-resources"
_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,99}$")


def redirect_uri() -> str:
    return get_settings().public_app_url.rstrip("/") + CALLBACK_PATH


def env_names(provider: OAuthProvider) -> list[str]:
    return [f"{provider.env.upper()}_CLIENT_ID", f"{provider.env.upper()}_CLIENT_SECRET"]


def missing_settings(key: str) -> list[str]:
    """Env vars that must be set before this vendor's OAuth flow can start."""
    p = PROVIDERS[key]
    s = get_settings()
    return [name for name, attr in zip(env_names(p), ("client_id", "client_secret"), strict=True) if not (getattr(s, f"{p.env}_{attr}", None) or "").strip()]


def _app(p: OAuthProvider) -> tuple[str, str]:
    missing = missing_settings(p.key)
    if missing:
        raise NotAvailable(f"The {p.vendor} OAuth app is not configured on this deployment. A platform operator must set {', '.join(missing)} (redirect URI {redirect_uri()}).", details={"needs_setup": missing})
    s = get_settings()
    return getattr(s, f"{p.env}_client_id").strip(), getattr(s, f"{p.env}_client_secret").strip()


def require_app(key: str) -> None:
    """Raise ``NotAvailable`` (listing the env vars) unless this vendor's OAuth app is configured."""
    _app(PROVIDERS[key])


def tenant_for(config: dict[str, Any]) -> str:
    tenant = (config.get("tenant") or get_settings().oauth_microsoft_tenant or "common").strip()
    if not _TENANT_RE.fullmatch(tenant):
        raise ValidationFailed("tenant must be a directory id, a verified domain, 'common' or 'organizations'.")
    return tenant


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)  # 86 chars, within RFC 7636's 43–128
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def nonce_hash(nonce: str) -> str:
    return hashlib.sha256(nonce.encode()).hexdigest()


def authorization_url(*, provider_key: str, scopes: list[str], org_id: uuid.UUID, user_id: uuid.UUID, integration_id: uuid.UUID, integration_provider: str, config: dict[str, Any]) -> tuple[str, str]:
    """Return (authorization URL, nonce). The caller stores ``nonce_hash(nonce)`` for single use."""
    p = PROVIDERS[provider_key]
    client_id, _ = _app(p)
    verifier, challenge = pkce_pair()
    nonce = secrets.token_urlsafe(18)
    tenant = tenant_for(config) if p.key == "microsoft" else None
    state = create_purpose_token(user_id=user_id, purpose=STATE_TYPE, minutes=STATE_MINUTES, extra={"org": str(org_id), "iid": str(integration_id), "prv": integration_provider, "op": p.key, "nonce": nonce, "pkv": encrypt_secret(verifier), **({"tnt": tenant} if tenant else {})})
    params = {"client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri(), "state": state, "code_challenge": challenge, "code_challenge_method": "S256", **p.auth_params}
    if scopes:
        params["scope"] = p.scope_sep.join(scopes)
    return p.auth_url.format(tenant=tenant or "common") + "?" + urlencode(params), nonce


def decode_state(state: str) -> dict[str, Any]:
    """Verify signature, expiry and type. Raises ``jwt.ExpiredSignatureError`` / ``jwt.InvalidTokenError``."""
    claims = decode_token(state, expected_type=STATE_TYPE)
    for k in ("org", "iid", "prv", "op", "nonce", "pkv"):
        if not claims.get(k):
            raise jwt.InvalidTokenError(f"state is missing {k}")
    if claims["op"] not in PROVIDERS:
        raise jwt.InvalidTokenError("unknown provider")
    return claims


def state_verifier(claims: dict[str, Any]) -> str:
    return decrypt_secret(claims["pkv"])


def _token_request(p: OAuthProvider, data: dict[str, str], *, tenant: str | None) -> dict[str, Any]:
    client_id, client_secret = _app(p)
    url = p.token_url.format(tenant=tenant or "common")
    auth = None
    if p.token_auth == "basic":
        auth = (client_id, client_secret)
    else:
        data = {**data, "client_id": client_id, "client_secret": client_secret}
    http = VendorHTTP(timeout=20.0, max_retries=2, max_wait=10.0)
    try:
        if p.token_format == "json":
            r = http.request("POST", url, json=data, auth=auth, headers={"Accept": "application/json"})
        else:
            r = http.request("POST", url, data=data, auth=auth, headers={"Accept": "application/json"})
    finally:
        http.close()
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code >= 400 or not isinstance(body, dict) or (p.key == "slack" and not body.get("ok")):
        reason = (body.get("error_description") or body.get("error")) if isinstance(body, dict) else None
        raise ConnectorError(f"{p.vendor} token endpoint refused the request: {str(reason)[:160]}." if reason else f"{p.vendor} token endpoint refused the request (HTTP {r.status_code}).")
    if not body.get("access_token"):
        raise ConnectorError(f"{p.vendor} returned no access token.")
    return body


def _normalise(p: OAuthProvider, body: dict[str, Any], *, previous: dict[str, Any] | None = None, requested: list[str] | None = None, tenant: str | None = None) -> dict[str, Any]:
    prev = previous or {}
    expires_in = body.get("expires_in")
    tokens: dict[str, Any] = {
        "type": "oauth",
        "provider": p.key,
        "access_token": body["access_token"],
        "refresh_token": body.get("refresh_token") or prev.get("refresh_token"),
        "expires_at": int(time.time()) + int(expires_in) if expires_in else None,
        "scope": body.get("scope") or prev.get("scope"),
        "requested_scopes": requested if requested is not None else prev.get("requested_scopes", []),
        "obtained_at": int(time.time()),
        "extras": dict(prev.get("extras") or {}),
    }
    if tenant:
        tokens["extras"]["tenant"] = tenant
    if p.key == "slack":
        team = body.get("team") or {}
        tokens["extras"].update({k: v for k, v in {"team_id": team.get("id"), "team_name": team.get("name"), "bot_user_id": body.get("bot_user_id"), "app_id": body.get("app_id")}.items() if v})
    elif p.key == "notion":
        tokens["extras"].update({k: v for k, v in {"workspace_id": body.get("workspace_id"), "workspace_name": body.get("workspace_name"), "bot_id": body.get("bot_id")}.items() if v})
    elif p.key == "dropbox":
        tokens["extras"].update({k: v for k, v in {"account_id": body.get("account_id")}.items() if v})
    return tokens


def exchange_code(*, provider_key: str, code: str, verifier: str, scopes: list[str], tenant: str | None = None) -> dict[str, Any]:
    p = PROVIDERS[provider_key]
    body = _token_request(p, {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri(), "code_verifier": verifier}, tenant=tenant)
    return _normalise(p, body, requested=scopes, tenant=tenant)


def complete(tokens: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Provider-specific post-exchange steps (Atlassian: find the Confluence cloud id)."""
    if tokens["provider"] != "atlassian":
        return tokens
    http = VendorHTTP(timeout=20.0)
    try:
        r = http.request("GET", ATLASSIAN_RESOURCES, headers={"Authorization": f"Bearer {tokens['access_token']}", "Accept": "application/json"})
    finally:
        http.close()
    if r.status_code >= 400:
        raise ConnectorError(vendor_error(r, "Atlassian"))
    sites = [s for s in r.json() if isinstance(s, dict) and s.get("id")]
    wanted = (config.get("site_url") or "").rstrip("/").lower()
    confluence = [s for s in sites if any("confluence" in sc for sc in s.get("scopes", []))] or sites
    pick = next((s for s in confluence if wanted and (s.get("url") or "").rstrip("/").lower() == wanted), None) or (confluence[0] if confluence and not wanted else None)
    if pick is None:
        raise ConnectorError("The authorized Atlassian account has no Confluence site" + (f" matching {wanted}." if wanted else "."))
    tokens["extras"].update({"cloud_id": pick["id"], "site_url": (pick.get("url") or "").rstrip("/"), "site_name": pick.get("name")})
    return tokens


def expired(tokens: dict[str, Any], skew: int = 60) -> bool:
    exp = tokens.get("expires_at")
    return bool(exp) and int(exp) - skew <= time.time()


def refresh(tokens: dict[str, Any]) -> dict[str, Any]:
    """Refresh-token grant. Rotated refresh tokens replace the old one; otherwise it is kept."""
    p = PROVIDERS[tokens["provider"]]
    if not tokens.get("refresh_token"):
        raise ConnectorError(f"The {p.vendor} authorization expired and cannot be refreshed; reconnect the integration.")
    tenant = (tokens.get("extras") or {}).get("tenant")
    data = {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]}
    if p.key == "microsoft" and tokens.get("requested_scopes"):
        data["scope"] = " ".join(tokens["requested_scopes"])
    try:
        body = _token_request(p, data, tenant=tenant)
    except ConnectorError as exc:
        raise ConnectorError(f"Refreshing the {p.vendor} authorization failed; reconnect the integration. ({exc.message})") from exc
    return _normalise(p, body, previous=tokens)


def parse(secret: str | None) -> dict[str, Any] | None:
    if not secret or not secret.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(secret)
    except ValueError:
        return None
    return data if isinstance(data, dict) and data.get("type") == "oauth" and data.get("access_token") else None


def dump(tokens: dict[str, Any]) -> str:
    return json.dumps(tokens, separators=(",", ":"))


def account_label(tokens: dict[str, Any] | None) -> str | None:
    if not tokens:
        return None
    x = tokens.get("extras") or {}
    return x.get("team_name") or x.get("workspace_name") or x.get("site_name") or (urlparse(x["site_url"]).netloc if x.get("site_url") else None)
