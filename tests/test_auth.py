from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from dopa_api.auth import AuthenticationError, SupabaseJwtVerifier


class _SigningKey:
    def __init__(self, key: object) -> None:
        self.key = key


class _JwksClient:
    def __init__(self, key: object) -> None:
        self.key = key

    def get_signing_key_from_jwt(self, _token: str) -> _SigningKey:
        return _SigningKey(self.key)


def _token(
    private_key: object,
    *,
    issuer: str,
    audience: str = "authenticated",
    role: str = "authenticated",
    expires_at: int | None = None,
) -> str:
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": "user-123",
            "role": role,
            "exp": expires_at or int(time.time()) + 300,
        },
        private_key,
        algorithm="ES256",
        headers={"kid": "test-key"},
    )


def test_verifies_es256_supabase_token() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    verifier = SupabaseJwtVerifier("https://example.supabase.co")
    verifier.jwks_client = _JwksClient(private_key.public_key())

    user = verifier.verify(_token(private_key, issuer=verifier.issuer))

    assert user.subject == "user-123"


@pytest.mark.parametrize(
    ("claims", "message"),
    [
        ({"issuer": "https://wrong.example/auth/v1"}, "Invalid or expired"),
        ({"audience": "anon"}, "Invalid or expired"),
        ({"role": "anon"}, "not an authenticated"),
        ({"expires_at": int(time.time()) - 30}, "Invalid or expired"),
    ],
)
def test_rejects_invalid_claims(claims: dict[str, object], message: str) -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    verifier = SupabaseJwtVerifier("https://example.supabase.co")
    verifier.jwks_client = _JwksClient(private_key.public_key())

    with pytest.raises(AuthenticationError, match=message):
        verifier.verify(
            _token(
                private_key,
                issuer=str(claims.get("issuer", verifier.issuer)),
                audience=str(claims.get("audience", "authenticated")),
                role=str(claims.get("role", "authenticated")),
                expires_at=(
                    int(claims["expires_at"]) if "expires_at" in claims else None
                ),
            )
        )
