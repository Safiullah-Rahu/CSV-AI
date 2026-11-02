"""Versioned API router (``/api/v1``)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import admin, ai, audit, auth, chat, command_center, databases, health, knowledge, orgs, platform, roles, structure

api_router = APIRouter()
for module in (health, auth, orgs, structure, roles, knowledge, chat, databases, command_center, ai, platform, audit, admin):
    api_router.include_router(module.router)
