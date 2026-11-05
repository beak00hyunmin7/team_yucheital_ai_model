"""
분석 기록 저장 (MySQL + 파일 저장소).

프런트(vtkjs.html)가 예측을 끝낼 때마다 기록 1건을 POST 로 보내면 서버에 저장한다.
예전에는 브라우저 localStorage 에만 있어서 다른 PC·브라우저에서는 기록이 안 보였다.

저장 위치
  - 기록 정보(이름, 조건, 배수구 좌표, 모델명 ...) -> MySQL `analysis_history` 테이블
  - 결과 그림(썸네일 JPEG)                         -> 파일: {HISTORY_DIR}/{사용자 폴더}/{id}.jpg
    DB 에는 그림 자체가 아니라 파일 경로만 넣는다 (DB 가 무거워지지 않게).

  YUCHE_HISTORY_DIR  썸네일 저장 폴더 (기본: webapp/data/history)

API (모두 `Authorization: Bearer <token>` 필요 = 로그인한 사용자 본인 기록만)
  GET    /api/history            최근 기록 목록 (최신순, 최대 50건, 썸네일 포함)
  POST   /api/history            기록 1건 저장 -> 저장된 기록 반환
  PATCH  /api/history/{id}       이름 변경  { "name": "..." }  (null/빈 문자열 = 기본 이름)
  DELETE /api/history/{id}       1건 삭제
  DELETE /api/history            내 기록 전체 삭제

테이블은 auth.py 와 같은 Base 를 쓰므로 서버 기동 시 auth.init_db() 가 함께 만든다.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, String, Text, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from auth import Base, SessionLocal, require_user

WEBAPP_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = Path(os.environ.get("YUCHE_HISTORY_DIR", WEBAPP_DIR / "data" / "history")).resolve()
MAX_RECORDS_LISTED = 50
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024  # 프런트가 긴 변 360px JPEG 로 줄여 보내므로 보통 50KB 안팎


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(String(32), primary_key=True)
    user_email = Column(String(255), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    name = Column(String(100), nullable=True)            # null 이면 프런트가 '날짜 시간' 표시
    file_name = Column(String(255), nullable=True)       # 사용자가 올린 입력 파일 이름
    params_json = Column(Text, nullable=True)            # {mode, rain_mm, time_s, dx}
    drain_json = Column(Text, nullable=True)             # {x, y} 또는 null
    checkpoint = Column(String(255), nullable=True)
    meta_json = Column(Text, nullable=True)
    info_json = Column(Text, nullable=True)              # 결과 안내문 리스트
    image_label = Column(String(100), nullable=True)
    thumbnail_path = Column(String(500), nullable=True)  # HISTORY_DIR 기준 상대 경로


# --------------------------------------------------------------------------- #
# 요청 형식
# --------------------------------------------------------------------------- #
class RecordIn(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    fileName: str | None = Field(default=None, max_length=255)
    params: dict[str, Any] | None = None
    drain: dict[str, Any] | None = None
    checkpoint: str | None = Field(default=None, max_length=255)
    meta: dict[str, Any] | None = None
    info: list[str] | None = None
    imageLabel: str | None = Field(default=None, max_length=100)
    thumbnail: str | None = None  # "data:image/jpeg;base64,...."


class RenameIn(BaseModel):
    name: str | None = Field(default=None, max_length=100)


# --------------------------------------------------------------------------- #
# 헬퍼
# --------------------------------------------------------------------------- #
def _get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(503, "기록 DB 에 연결할 수 없습니다.") from exc
    finally:
        db.close()


def _user_dir(email: str) -> Path:
    # 이메일을 그대로 폴더 이름으로 쓰지 않는다 (특수문자·개인정보 노출 방지)
    return HISTORY_DIR / hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]


def _save_thumbnail(email: str, record_id: str, data_url: str | None) -> str | None:
    if not data_url:
        return None
    if "," not in data_url or not data_url.startswith("data:image/"):
        raise HTTPException(422, "thumbnail 은 data:image/...;base64, 형식이어야 합니다.")
    header, b64 = data_url.split(",", 1)
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "thumbnail base64 를 해석할 수 없습니다.") from exc
    if len(raw) > MAX_THUMBNAIL_BYTES:
        raise HTTPException(413, "thumbnail 이 너무 큽니다 (최대 2MB).")
    ext = "png" if "image/png" in header else "jpg"

    folder = _user_dir(email)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record_id}.{ext}"
    path.write_bytes(raw)
    return str(path.relative_to(HISTORY_DIR))


def _thumbnail_data_url(rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    path = (HISTORY_DIR / rel_path).resolve()
    if not str(path).startswith(str(HISTORY_DIR)) or not path.is_file():
        return None
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _delete_thumbnail(rel_path: str | None) -> None:
    if not rel_path:
        return
    path = (HISTORY_DIR / rel_path).resolve()
    if str(path).startswith(str(HISTORY_DIR)) and path.is_file():
        path.unlink()


def _loads(text: str | None):
    return json.loads(text) if text else None


def _dumps(value) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def _to_dict(r: AnalysisHistory) -> dict:
    """프런트가 쓰던 localStorage 기록과 같은 모양으로 돌려준다 (프런트 렌더 코드 그대로 사용)."""
    created = r.created_at.isoformat() if r.created_at else None
    return {
        "id": r.id,
        "createdAt": created,
        "name": r.name,
        "fileName": r.file_name,
        "params": _loads(r.params_json),
        "drain": _loads(r.drain_json),
        "checkpoint": r.checkpoint,
        "meta": _loads(r.meta_json),
        "info": _loads(r.info_json) or [],
        "imageLabel": r.image_label,
        "thumbnail": _thumbnail_data_url(r.thumbnail_path),
    }


def _own_record(db: Session, record_id: str, email: str) -> AnalysisHistory:
    r = db.get(AnalysisHistory, record_id)
    if r is None or r.user_email != email:   # 남의 기록은 '없음'으로 응답
        raise HTTPException(404, "기록을 찾을 수 없습니다.")
    return r


# --------------------------------------------------------------------------- #
# 라우트
# --------------------------------------------------------------------------- #
router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("")
def list_history(email: str = Depends(require_user), db: Session = Depends(_get_db)) -> dict:
    rows = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.user_email == email)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(MAX_RECORDS_LISTED)
        .all()
    )
    return {"records": [_to_dict(r) for r in rows]}


@router.post("")
def create_history(
    body: RecordIn, email: str = Depends(require_user), db: Session = Depends(_get_db)
) -> dict:
    record_id = uuid.uuid4().hex
    thumb = _save_thumbnail(email, record_id, body.thumbnail)
    r = AnalysisHistory(
        id=record_id,
        user_email=email,
        name=(body.name or "").strip() or None,
        file_name=body.fileName,
        params_json=_dumps(body.params),
        drain_json=_dumps(body.drain),
        checkpoint=body.checkpoint,
        meta_json=_dumps(body.meta),
        info_json=_dumps(body.info),
        image_label=body.imageLabel,
        thumbnail_path=thumb,
    )
    try:
        db.add(r)
        db.commit()
    except SQLAlchemyError:
        _delete_thumbnail(thumb)   # DB 저장 실패 시 파일만 남지 않게
        raise
    db.refresh(r)
    return _to_dict(r)


@router.patch("/{record_id}")
def rename_history(
    record_id: str, body: RenameIn,
    email: str = Depends(require_user), db: Session = Depends(_get_db),
) -> dict:
    r = _own_record(db, record_id, email)
    r.name = (body.name or "").strip() or None
    db.commit()
    db.refresh(r)
    return _to_dict(r)


@router.delete("/{record_id}")
def delete_history(
    record_id: str, email: str = Depends(require_user), db: Session = Depends(_get_db)
) -> dict:
    r = _own_record(db, record_id, email)
    thumb = r.thumbnail_path
    db.delete(r)
    db.commit()
    _delete_thumbnail(thumb)
    return {"ok": True}


@router.delete("")
def clear_history(email: str = Depends(require_user), db: Session = Depends(_get_db)) -> dict:
    rows = db.query(AnalysisHistory).filter(AnalysisHistory.user_email == email).all()
    thumbs = [r.thumbnail_path for r in rows]
    for r in rows:
        db.delete(r)
    db.commit()
    for t in thumbs:
        _delete_thumbnail(t)
    return {"ok": True, "deleted": len(rows)}
