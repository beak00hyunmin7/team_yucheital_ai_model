"""
회원가입 / 로그인 (MySQL + bcrypt).

기존 sejong_yucheital_project/.../yuche_backend 의 로직을 웹앱 한 서버 안으로 옮긴 것.
DB 는 MySQL 그대로 사용. 접속 정보는 환경변수로 덮어쓸 수 있다.

  YUCHE_DB_URL   기본 mysql+pymysql://root:1234@127.0.0.1:3306/yuche_db
  YUCHE_SECRET   로그인 토큰 서명 키. 안 주면 webapp/.yuche_secret 에 임의값을 만들어 쓴다

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
import secrets
import subprocess
import sys
import time

import re

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, create_engine, func
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DB_URL = os.environ.get(
    "YUCHE_DB_URL", "mysql+pymysql://root:1234@127.0.0.1:3306/yuche_db?charset=utf8mb4"
)
def _load_secret() -> bytes:
    """토큰 서명 키. 환경변수 > 로컬 파일 > (쓰기 실패 시) 메모리 전용 임의값.

    예전 기본값은 "dev-only-change-me" 고정이었다. 127.0.0.1 전용일 때는 넘어갈 수 있지만
    랜에 열어두는 순간 위험해진다 - 이 저장소는 공개라 키가 그대로 적혀 있고, 서명 키를
    아는 사람은 아무 이메일로나 유효한 토큰을 **가입 없이** 만들어 낼 수 있다.
    (verify_token 은 서명과 만료만 보고 DB 를 조회하지 않는다.)

    그래서 기본값을 없애고 기기마다 다른 임의값을 쓴다. 파일로 남기므로 서버를 재시작해도
    이미 로그인한 사람이 튕기지 않는다. 키를 바꾸고 싶으면 .yuche_secret 을 지우면 된다.
    """
    env = os.environ.get("YUCHE_SECRET")
    if env:
        return env.encode()

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".yuche_secret")
    try:
        if os.path.exists(path):
            saved = open(path, encoding="ascii").read().strip()
            if saved:
                return saved.encode()
        value = secrets.token_urlsafe(48)
        with open(path, "w", encoding="ascii") as f:
            f.write(value)
        return value.encode()
    except OSError:
        # 파일을 못 쓰는 환경이면 프로세스 안에서만 유효한 키를 쓴다.
        # (재시작하면 기존 토큰이 전부 무효가 되지만, 고정값을 쓰는 것보다 낫다.)
        return secrets.token_urlsafe(48).encode()


SECRET = _load_secret()
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


# --- DB 연결 자동 복구 -------------------------------------------------------
# 회원가입이 "서버는 켜져 있지만 요청을 처리하다 오류" 로 실패하던 원인: 가입·로그인 세션은
# 요청을 처리하는 도중에 처음 DB 연결을 여는데, 그 순간 Windows -> WSL 127.0.0.1:3306 포워딩이
# 끊겨 있으면(WinError 10061) 아무 복구 없이 500 이 났다. /api/health 가 ok 인 직후에도 일어난다.
# 이 포워딩은 Windows 에서 `wsl` 명령을 한 번 날리면 되살아나므로(run_windows.bat 의 유지용 루프와
# 같은 원리), 연결에 실패하면 WSL 을 한 번 깨우고 다시 연결해 본다. 그래도 안 되면 500 이 아니라
# 사람이 읽을 수 있는 503 으로 알린다.
_WSL_DISTRO = os.environ.get("YUCHE_WSL_DISTRO", "Ubuntu-22.04")
_POKE_COOLDOWN_S = 5.0
_last_poke = 0.0


def _poke_wsl() -> None:
    """WSL 을 깨워 포워딩을 되살린다 (Windows 전용, 5초에 한 번만)."""
    global _last_poke
    if sys.platform != "win32" or time.time() - _last_poke < _POKE_COOLDOWN_S:
        return
    _last_poke = time.time()
    try:
        subprocess.run(["wsl", "-d", _WSL_DISTRO, "-u", "root", "--", "true"],
                       timeout=20, capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001  (wsl 이 없거나 느려도 요청 자체는 아래에서 처리)
        pass


def connect_with_recovery(db: Session, attempts: int = 3) -> None:
    """세션의 DB 연결을 지금 열고, 거부되면 WSL 을 깨워 몇 번 다시 시도한다."""
    for i in range(attempts):
        try:
            db.connection()
            return
        except OperationalError:
            db.rollback()
            if i == attempts - 1:
                break
            if i == 0:
                _poke_wsl()
            time.sleep(1.0 * (i + 1))
    raise HTTPException(
        503,
        "회원 DB(MariaDB)에 연결할 수 없습니다. WSL 이 꺼져 있거나 연결이 끊긴 상태예요. "
        "잠시 후 다시 시도하고, 계속되면 서버를 run_windows.bat 로 다시 켜 주세요.",
    )


def _get_db():
    db = SessionLocal()
    try:
        connect_with_recovery(db)
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
    # 공백을 뗀 이메일 하나로 검사·저장한다. 예전엔 조회는 원본, 저장은 strip() 본이라
    # ' a@b.com' 으로 가입하면 중복 검사를 빠져나가 UNIQUE 위반(500)이 났다.
    email = body.email.strip()
    if not _EMAIL_RE.match(email):
        raise HTTPException(400, "이메일 형식이 올바르지 않습니다.")
    if len(body.password) < 6:
        raise HTTPException(400, "비밀번호는 6자 이상이어야 합니다.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "이미 가입된 이메일입니다.")
    user = User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:   # 같은 이메일로 동시에 가입한 경우
        db.rollback()
        raise HTTPException(409, "이미 가입된 이메일입니다.") from None
    return {"ok": True, "email": user.email, "token": make_token(user.email)}


@router.post("/login")
def login(body: Credentials, db: Session = Depends(_get_db)):
    user = db.query(User).filter(User.email == body.email.strip()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "이메일 또는 비밀번호가 올바르지 않습니다.")
    return {"ok": True, "email": user.email, "token": make_token(user.email)}


@router.get("/me")
def me(email: str = Depends(require_user)):
    return {"email": email}
