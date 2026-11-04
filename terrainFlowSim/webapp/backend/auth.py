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


# DB 준비 상태는 "기동 때 한 번" 판정하면 안 된다.
#
# 이 환경의 MariaDB 는 WSL2 안에 있고 Windows -> 127.0.0.1:3306 포워딩이 수시로 끊긴다
# (연속 8/8 성공 직후 바로 연결거부(WinError 10061)를 관측). 끊김은 방향이 둘인데
# 기존 코드는 한쪽만 막혀 있었다.
#   (1) 기동 때 살아 있다가 뒤에 끊김  -> app._auth_gate 가 처리한다 (토큰 검증 실패 시 통과).
#   (2) 기동 때 끊겨 있다가 뒤에 살아남 -> 처리되지 않았다. init_db() 의 False 가 그대로
#       latch 되어 프로세스가 죽을 때까지 인증이 꺼진 채 고정된다. 실제로 이 상태로 서버가
#       떴고, 회원가입·로그인이 조용히 비활성인 것을 /api/health 로만 알 수 있었다.
#
# 그래서 짧은 TTL 캐시로 매번 다시 판정한다. 처음 연결된 순간에 테이블을 만들므로
# DB 가 늦게 떠도 가입이 바로 된다.
_READY_TTL_S = 5.0            # 요청마다 TCP 연결을 열지 않도록 이 시간만 캐시
_ready_cache: tuple[float, bool] = (0.0, False)
_tables_created = False
_last_error = ""


def db_ready(force: bool = False) -> bool:
    """DB 를 지금 쓸 수 있는지. 실패 이유는 db_error() 로 남긴다."""
    global _ready_cache, _tables_created, _last_error
    now = time.time()
    if not force and now - _ready_cache[0] < _READY_TTL_S:
        return _ready_cache[1]

    ok = False
    try:
        if not _tables_created:
            Base.metadata.create_all(bind=engine)   # 연결 + 최초 1회 테이블 생성
            _tables_created = True
        else:
            with engine.connect() as c:
                c.exec_driver_sql("SELECT 1")
        ok = True
        _last_error = ""
    except (OperationalError, SQLAlchemyError) as e:
        # 기존 코드는 예외를 통째로 버려서 "db: down" 의 이유를 알 방법이 없었다.
        # pymysql 이 Windows 오류 메시지를 CP949 로 잘못 디코드해서 깨진 글자가 섞여
        # 온다. JSON 으로 못 내보내는 서러게이트만 걷어낸다 (WinError 번호는 남는다).
        msg = " ".join(str(getattr(e, "orig", e)).splitlines())
        _last_error = "".join(c for c in msg if not 0xD800 <= ord(c) <= 0xDFFF)[:200]
    _ready_cache = (now, ok)
    return ok


def init_db() -> bool:
    """기동 시 1회 호출용. 반환값을 상수로 들고 있지 말고 db_ready() 를 그때그때 쓸 것."""
    return db_ready(force=True)


def db_status() -> str:
    return "ok" if db_ready() else "down"


def db_error() -> str:
    """마지막 연결 실패 이유 (연결되어 있으면 빈 문자열)."""
    return _last_error


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
