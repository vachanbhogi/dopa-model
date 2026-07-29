"""Supabase access-token verification for the Dopa API."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
from jwt import PyJWKClient

DEFAULT_SUPABASE_URL = "https://bdqieraaueobnrjcxwtf.supabase.co"


class AuthenticationError(Exception):
    """Raised when an access token cannot authenticate a Supabase user."""


@dataclass(frozen=True)
class AuthenticatedUser:
    subject: str


class SupabaseJwtVerifier:
    """Verify Supabase ES256 access tokens against the project's public JWKS."""

    def __init__(self, supabase_url: str | None = None) -> None:
        self.supabase_url = (
            supabase_url or os.environ.get("DOPA_SUPABASE_URL") or DEFAULT_SUPABASE_URL
        ).rstrip("/")
        self.issuer = f"{self.supabase_url}/auth/v1"
        self.jwks_client = PyJWKClient(
            f"{self.issuer}/.well-known/jwks.json",
            cache_keys=True,
            lifespan=600,
        )

    def verify(self, token: str) -> AuthenticatedUser:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "ES256":
                raise AuthenticationError("Unsupported token algorithm.")
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",
                issuer=self.issuer,
                leeway=5,
                options={"require": ["aud", "exp", "iss", "sub"]},
            )
        except AuthenticationError:
            raise
        except jwt.PyJWTError as error:
            raise AuthenticationError("Invalid or expired access token.") from error
        except Exception as error:
            raise AuthenticationError("Unable to verify access token.") from error

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("Access token is missing its subject.")
        if claims.get("role") != "authenticated":
            raise AuthenticationError(
                "Access token is not an authenticated user token."
            )
        return AuthenticatedUser(subject=subject)
