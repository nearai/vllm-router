import os
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
USER_TOKEN = os.environ.get("USER_TOKEN")

if not ADMIN_TOKEN:
    raise RuntimeWarning("ADMIN_TOKEN is not set. Admin access will not be available.")
if not USER_TOKEN:
    raise RuntimeWarning("USER_TOKEN is not set. User access will not be available.")


def verify_admin_access(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_TOKEN is not configured on the server",
        )
    if token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def verify_user_access(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not USER_TOKEN and not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication tokens are not configured on the server",
        )

    is_admin = ADMIN_TOKEN and token == ADMIN_TOKEN
    is_user = USER_TOKEN and token == USER_TOKEN

    if not (is_admin or is_user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
