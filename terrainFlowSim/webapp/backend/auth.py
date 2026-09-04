"""
회원가입 / 로그인 (MySQL + bcrypt).

기존 sejong_yucheital_project/.../yuche_backend 의 로직을 웹앱 한 서버 안으로 옮긴 것.
DB 는 MySQL 그대로 사용. 접속 정보는 환경변수로 덮어쓸 수 있다.

  YUCHE_DB_URL   기본 mysql+pymysql://root:1234@127.0.0.1:3306/yuche_db
  YUCHE_SECRET   로그인 토큰 서명 키 (기본은 개발용 고정값 — 배포 시 반드시 교체)

로그인하면 HMAC 서명 토큰을 돌려주고, 프런트가 그걸 저장했다가
`Authorization: Bearer <token>` 로 /api/predict 를 호출한다.
(원본과 달리 예측 API 를 로그인 뒤에서만 쓰게 막았다.)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

import re

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, create_engine, func
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_URL = os.environ.get(
    "YUCHE_DB_URL", "mysql+pymysql://root:1234@127.0.0.1:3306/yuche_db?charset=utf8mb4"
)
SECRET = os.environ.get("YUCHE_SECRET", "dev-only-change-me").encode()
TOKEN_TTL = 60 * 60 * 12  # 12시간

engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=280)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db() -> bool:
    """테이블 생성. DB 가 안 떠 있으면 False (앱은 계속 뜨고 인증만 비활성)."""
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except (OperationalError, SQLAlchemyError):
        return False


def db_status() -> str:
    try:
        with engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return "ok"
    except Exception:  # noqa: BLE001
        return "down"


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 비밀번호 ---
def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("ascii")


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode("ascii"))
    except ValueError:
        return False


# --- 토큰 (HMAC 서명, 의존성 없음) ---
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(email: str) -> str:
    payload = _b64(json.dumps({"email": email, "exp": int(time.time()) + TOKEN_TTL}).encode())
    sig = _b64(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify_token(token: str) -> str | None:
    try:
        payload, sig = token.split(".", 1)
        expected = _b64(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_unb64(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data.get("email")
    except Exception:  # noqa: BLE001
        return None


def require_user(authorization: str = Header(default="")) -> str:
    """/api/predict 등 보호 라우트용 의존성. 유효 토큰의 이메일을 반환."""
    token = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    email = verify_token(token) if token else None
    if not email:
        raise HTTPException(401, "로그인이 필요합니다.")
    return email


# --- 라우트 ---
router = APIRouter(prefix="/api/auth", tags=["auth"])


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str
    password: str


@router.post("/signup")
def signup(body: Credentials, db: Session = Depends(_get_db)):
    if not _EMAIL_RE.match(body.email.strip()):
        raise HTTPException(400, "이메일 형식이 올바르지 않습니다.")
    if len(body.password) < 6:
        raise HTTPException(400, "비밀번호는 6자 이상이어야 합니다.")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(409, "이미 가입된 이메일입니다.")
    user = User(email=body.email.strip(), password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"ok": True, "email": user.email, "token": make_token(user.email)}


@router.post("/login")
def login(body: Credentials, db: Session = Depends(_get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "이메일 또는 비밀번호가 올바르지 않습니다.")
    return {"ok": True, "email": user.email, "token": make_token(user.email)}


@router.get("/me")
def me(email: str = Depends(require_user)):
    return {"email": email}
