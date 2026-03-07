"""JWT Authentication using Supabase tokens."""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from . import settings

security = HTTPBearer()


def decode_token(token: str) -> dict:
    """Decode and verify Supabase JWT."""
    try:
        return jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Get current user from JWT. Returns decoded payload."""
    return decode_token(credentials.credentials)


async def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Require admin role from JWT app_metadata."""
    payload = decode_token(credentials.credentials)
    role = (payload.get("app_metadata") or {}).get("role")
    if role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return payload
