import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from auth import service

logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str


auth_router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)


@auth_router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest) -> AuthResponse:
    """Login with email and password."""
    try:
        result = service.login_user(request.email, request.password)
        if result:
            logger.info(f"User {request.email} logged in successfully")
            return AuthResponse(success=True, message="Login successful")
        else:
            logger.warning(f"Failed login attempt for {request.email}")
            return AuthResponse(success=False, message="Invalid email or password")
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@auth_router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest) -> AuthResponse:
    """Register a new user with email and password."""
    try:
        result = service.register_user(request.email, request.password)
        if result:
            logger.info(f"User {request.email} registered successfully")
            return AuthResponse(success=True, message="Registration successful")
        else:
            logger.warning(f"Registration failed for {request.email}")
            return AuthResponse(success=False, message="Registration failed")
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

