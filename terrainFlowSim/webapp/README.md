# 예측 배수맵 웹 (webapp)

등고선을 넣으면 학습된 AI 모델(`ai_model/`)이 예측 배수맵(지표 유출 집중도)을 돌려주는 웹.

```
webapp/
├── backend/
│   ├── app.py            FastAPI. /api/predict + 인증 라우트 마운트
│   ├── auth.py           회원가입/로그인 (MySQL + bcrypt + HMAC 토큰)
│   ├── terrain_io.py     '등고선 데이터'(고도 격자) 파일 파서 (.npy/.csv/.asc/.tif ...)
│   └── requirements.txt  웹 서버 + 인증 의존성 (torch 등은 ai_model/.venv 재사용)
├── frontend/
│   └── index.html        로그인 → 입력 방식·강우량 선택 UI (단일 파일, 프레임워크 없음)
└── run_windows.bat       더블클릭 실행 (WSL MariaDB 시작 포함)
```

## 실행

전제: `ai_model/.venv` 존재 + WSL Ubuntu 에 MariaDB 설치됨(아래 참고).

```bat
webapp\run_windows.bat
```

또는 수동:

```powershell
wsl -d Ubuntu-22.04 -u root -- service mariadb start
cd project\terrainFlowSim\webapp
..\ai_model\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
..\ai_model\.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

브라우저에서 <http://127.0.0.1:8000> · API 문서 <http://127.0.0.1:8000/docs>

## 회원가입 / 로그인 (MySQL)

- DB 는 **WSL Ubuntu 안의 MariaDB**(MySQL 호환). Windows 쪽 Python 이 `localhost:3306` 으로 접속.
- 접속정보: `mysql+pymysql://root:1234@127.0.0.1:3306/yuche_db` — 환경변수 `YUCHE_DB_URL` 로 교체 가능.
- 로그인 토큰 서명키: 환경변수 `YUCHE_SECRET` (배포 시 반드시 교체).
- API: `POST /api/auth/signup`, `POST /api/auth/login` → `{ok, email, token}` / `GET /api/auth/me`.
  `POST /api/predict` 는 `Authorization: Bearer <token>` 필요 (DB 미연결 시 자동으로 인증 생략).
- 프런트: 첫 화면이 로그인, 성공하면 토큰을 localStorage 에 저장하고 예측 화면 표시.

### MariaDB 최초 설치 (WSL, 1회)

```bash
wsl -d Ubuntu-22.04 -u root -- bash -lc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y mariadb-server
  sed -i "s/^bind-address.*/bind-address = 0.0.0.0/" /etc/mysql/mariadb.conf.d/50-server.cnf
  service mariadb start
  mysql -uroot -e "
    CREATE USER IF NOT EXISTS \"root\"@\"%\" IDENTIFIED BY \"1234\";
    GRANT ALL ON *.* TO \"root\"@\"%\" WITH GRANT OPTION;
    CREATE DATABASE IF NOT EXISTS yuche_db CHARACTER SET utf8mb4;
    FLUSH PRIVILEGES;"
'
```

`users` 테이블(id, email, password_hash, created_at)은 서버 첫 기동 시 자동 생성.
WSL/PC 를 재시작하면 `service mariadb start` 를 다시 해야 함 (run_windows.bat 이 대신 해줌).

## 결과 그림

- **등고선 데이터 모드**: 예측 배수맵을 지도 스타일로 표출
  (`analysis_figure.py`, `visualize_inference.py` 이식) — 실제 등고선(m 라벨) +
  음영기복 + 축척막대 + 방위표 + **배수구 설치 후보**(예측 유출 최대점) 별표.
  응답에 `analysis_png_base64`, `drain_candidate{pixel, normalized}` 추가.
- **등고선 이미지 모드**: 원본 고도값이 없어 이 지도는 못 만들고, 기존 3장
  (입력 등고선 / 예측 배수맵 / 오버레이)만 반환.
- 한글 라벨은 `Malgun Gothic` 폰트 사용 (Windows 기본). 없으면 fallback.

## 입력 방식 2가지 (사용자가 화면에서 선택)

| 방식 | 보내는 것 | 서버 처리 | 언제 |
|---|---|---|---|
| **등고선 이미지** | PNG/JPG 등 | 그대로 모델 입력 | 이미 matplotlib `terrain` 스타일로 렌더된 등고선이 있을 때 |
| **등고선 데이터** | 고도 격자 파일 | 학습과 동일한 `render_contour_rgb` 로 등고선 이미지를 만든 뒤 모델 입력 | 원본 DEM/격자를 가지고 있을 때 (스타일 불일치가 없어 더 정확) |

'등고선 데이터' 지원 포맷 (`terrain_io.py` `_PARSERS`):
`.npy` · `.csv` `.txt` `.dat` (숫자 격자) · `.asc` `.grd` (ESRI ASCII Grid, cellsize 자동 인식) ·
`.tif` `.tiff` (단일 밴드) · `.png` `.jpg` (회색조 높이맵).
→ 팀에서 쓰는 실제 포맷이 정해지면 `_PARSERS` 에 함수 하나만 추가.

## 강우 조건

기본 체크포인트는 **FiLM 강우 조건 모델** (`ai_model/checkpoints_film/best.pt`).
화면의 강우량(mm) 슬라이더 값이 `rain_mm` 으로 전달돼 예측이 실제로 달라진다.
학습이 다룬 범위는 20~80mm.

## 모델 교체

```powershell
$env:AI_MODEL_CHECKPOINT = "C:\...\ai_model\checkpoints_gan\best.pt"   # 후 재시작
```

강우 조건이 없는 체크포인트를 넣으면 서버가 자동 감지해서 `rain_mm` 을 무시하고,
화면에도 "강우 조건 미반영"으로 표시된다. (`current_uses_film()` 자동 판별)

## 응답 (`POST /api/predict`, multipart)

폼 필드: `mode`(image|terrain), `rain_mm`, `dx`(terrain 전용), `file`

```json
{
  "contour_png_base64":   "…",   // 실제로 모델에 들어간 등고선
  "prediction_png_base64":"…",   // 예측 배수맵
  "overlay_png_base64":   "…",   // 등고선 + 예측 겹침
  "checkpoint": "best.pt",
  "meta": { "mode": "...", "grid_shape": [...], "rain_mm": 45.0, ... }
}
```

## 한계

- 예측은 지형수문 근사 시뮬레이터로 만든 학습쌍을 흉내낸 것. CFD/실측이 아님.
- 입력 등고선이 학습 데이터 스타일에서 멀수록(손그림·일반 지도 등) 정확도 급락.
- 해상도 256×256 고정. 큰 격자는 서버가 자동 축소 후 처리.
