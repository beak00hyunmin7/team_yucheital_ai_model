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

## 같은 공유기의 다른 PC·폰에서 접속

`run_windows.bat` 은 `0.0.0.0` 에 바인딩하므로 서버 쪽은 준비돼 있다. 기동하면 접속 주소를
직접 찍어준다. 프런트는 `const API = location.origin` 이라 다른 호스트로 들어와도 그대로 돈다.

남은 건 **방화벽 1회 허용**이다. 관리자 PowerShell 에서 한 번만:

```powershell
New-NetFirewallRule -DisplayName "yuche webapp 8000" -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8000 -Profile Private
```

`-Profile Private` 이 핵심이다. 공유기 네트워크가 Windows 에서 "공용"으로 잡혀 있으면 규칙이
안 먹으므로, 설정 > 네트워크에서 해당 연결을 **개인 네트워크**로 바꿔야 한다.

되돌릴 때: `Remove-NetFirewallRule -DisplayName "yuche webapp 8000"`
이 PC 에서만 쓰려면 `run_windows.bat` 의 `set BIND=0.0.0.0` 을 `127.0.0.1` 로.

### 열기 전에 알아둘 것

- **HTTP 평문이다.** 로그인 비밀번호가 같은 네트워크에서 그대로 보인다. 공용 Wi-Fi 에서는
  진짜 비밀번호를 쓰지 말 것.
- 토큰 서명 키는 이제 기기마다 임의 생성된다(`webapp/.yuche_secret`, git 제외).
  예전 고정 기본값 `dev-only-change-me` 는 이 저장소가 공개라 키가 그대로 적혀 있었고,
  서명 키를 아는 사람은 **가입 없이** 아무 이메일로나 유효한 토큰을 만들 수 있었다
  (`verify_token` 은 서명과 만료만 보고 DB 를 조회하지 않는다). 랜에 여는 이상 이건 막아야 한다.
- 인터넷 전체에 공개하는 건 다른 얘기다. 위 두 가지(HTTPS, 토큰 설계) 먼저 손봐야 하고,
  포트포워딩 대신 Tailscale 이나 Cloudflare Tunnel 을 쓰는 쪽이 훨씬 안전하다.

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

`users` 테이블(id, email, password_hash, created_at)은 DB 에 처음 연결되는 순간 자동 생성.
WSL/PC 를 재시작하면 `service mariadb start` 를 다시 해야 함 (run_windows.bat 이 대신 해줌).

### WSL2 가 저절로 꺼져서 DB 가 끊기는 문제 (2026-11-04 해결)

증상은 "조금 전까지 되던 로그인이 몇 분 뒤 안 된다" 였다. 원인은 DB 가 아니라 **WSL2 VM
자체가 유휴 상태에서 종료되는 것**이다. VM 이 내려가면 그 안의 MariaDB 도 같이 죽는다.

실측: `wsl` 명령 직후에는 `127.0.0.1:3306` 연결이 5/5 성공하고, 1~2분 방치하면 20/20
연결거부(WinError 10061)가 난다. 그 상태에서 WSL 에 들어가 보면 `uptime` 이 "up 0 minutes"
— VM 이 방금 새로 뜬 것이다. MariaDB 는 systemd 에 enabled 되어 있어 VM 이 뜰 때 자동
기동되므로 "WSL 을 건드리면 되살아나는" 것처럼 보여서 포워딩 문제로 오해하기 쉽다.

대책 두 가지를 같이 넣었다.

1. `run_windows.bat` 이 서버가 도는 동안 **Windows 쪽에서 45 초마다 `wsl ... true` 를 날리는
   WSL 유지용 루프**를 띄운다 (`yuche-wsl-keepalive` 창, 서버 종료 시 `taskkill` 로 정리).
   처음에는 WSL 안쪽에 `sleep infinity` 를 띄우는 방식이었는데, 그것만으로는 부족하다 —
   VM 과 mariadb 가 멀쩡히 살아 있는데도 Windows → WSL 의 `127.0.0.1:3306` **포워딩이
   따로 끊기는** 두 번째 고장 모드가 있다 (2026-11-05 재현). Windows 에서 `wsl` 명령을
   한 번 날리면 포워딩이 되살아나므로, 바깥에서 주기적으로 찔러 두 고장 모드를 함께 막는다.
2. `auth.db_ready()` 가 DB 준비 상태를 **그때그때 다시 판정**한다(5초 캐시). 기동 시점의
   판정을 상수로 들고 있으면, 기동 때 DB 가 꺼져 있었다는 이유만으로 그 뒤 DB 가 살아나도
   프로세스가 죽을 때까지 인증이 비활성으로 고정된다. 실제로 그 상태로 서버가 떠서
   회원가입·로그인이 조용히 죽어 있었다.

`/api/health` 의 `db_error` 로 끊긴 이유를 바로 확인할 수 있다 (예전에는 예외를 통째로
버려서 "down" 말고는 알 수 있는 게 없었다).

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
| **등고선 이미지** | PNG/JPG 등 | `terrain` 색 체계면 그대로, 아니면 색을 높이로 복원해 `terrain` 으로 다시 그려 모델 입력 (아래) | 등고선 이미지만 있을 때 |
| **등고선 데이터** | 고도 격자 파일 | 학습과 동일한 `render_contour_rgb` 로 등고선 이미지를 만든 뒤 모델 입력 | 원본 DEM/격자를 가지고 있을 때 (스타일 불일치가 없어 더 정확) |

### 이미지 색 체계와 높낮이 반전 (`contour_color.py`)

모델은 **색을 곧 높이**로 읽는다(학습 형식: 파랑=낮음 → 흰색=높음). 회색조·viridis·jet 처럼
다른 색 체계나 거꾸로 칠한 이미지를 그대로 넣으면 높낮이가 뒤집혀 예측된다.
그래서 화면에서 **색 체계**(자동 감지 / terrain·회색조·viridis·jet·gist_earth·rainbow·turbo·cividis)와
**높낮이 반대로 해석** 체크를 고르게 하고, 서버가 색→높이(0~1)를 복원해 `render_contour_rgb` 로
학습 형식으로 다시 그린다. 결과 화면에 원본·변환본·해석한 색 범례를 같이 보여준다.

- 자동 감지는 **컬러맵 종류**만 찾는다. 회색조나 거꾸로 칠한 지도는 색만으로 방향을 알 수 없어
  기본을 "밝을수록 높음"으로 가정하고 경고를 띄운다 → 반대면 체크박스.
- 이미 `terrain` 체계면 원본을 건드리지 않는다(학습 샘플 300장 중 오인 0건).
- **일반 지형도·토지피복도는 거부한다(자동 감지일 때 422).** 색이 높이가 아니라 숲·시가지·물 같은
  지물을 뜻하는 이미지(예: USGS 지형도)는 색이 우연히 `terrain` 과 94% 맞아도 높이를 읽을 수 없어
  예측이 의미 없는 얼룩이 된다. 색→높이로 읽은 값이 이웃끼리 튀는 비율(`_roughness`)로 판별한다:
  학습 형식 ≤0.028(JPEG70·확대해도 ≤0.016), 그 지형도 0.16. 기준 0.07. 색 체계를 직접 고른
  경우는 막지 않고 경고만 띄운다. 이런 지도는 DEM(고도 격자)을 '등고선 데이터' 모드로 올려야 한다.
- 검증(고도 격자 10개, 학습 형식 입력 예측과의 상관): 변환 전 회색조 0.53·jet 0.74·viridis 0.61·
  거꾸로 jet 0.31 → 변환 후 0.95~0.97.
- **무지개 색상(hue) 읽기**: 음영기복이 겹쳐 밝기가 변한 무지개색 지도(NOAA 수심도 등)는 RGB 최근접
  매칭이 23~36% 밖에 못 맞춘다. 이때는 색상(파랑=낮음→빨강=높음)으로 읽는 `hue` 체계로 한 번 더 본다
  (색 있는 픽셀 ≥50% · 색상이 넓게 퍼짐 · 이웃이 매끈할 때만 인정). 직접 고를 수도 있다.
- **어떤 색 체계와도 75% 미만으로 맞으면 거부(자동 감지일 때 422).** 예전엔 경고만 띄우고 `terrain`
  으로 가정해 통과시켜서 스캔 도면·3D 투시도·토지피복 지도도 말없이 얼룩 예측이 나왔다
  (기준: 학습 샘플·JPEG 최소 0.87, 합성 컬러맵 최소 0.84).
- **인터넷 예시 8장 실측 (Wikimedia Commons, 퍼블릭 도메인)**: 쓸 수 있는 입력은 2장뿐이었다 —
  회색조 높이맵(밝을수록 높음, 정상 변환·예측)과 무지개색 수심도(hue 로 변환). 지형도·스캔
  도면·3D 투시도 4장은 거부. 음영기복(hillshade)과 흑백 등고선 선 도면은 회색조라 거부하지 못해
  경고만 띄운다(밝기가 높이가 아니라 빛·선이라 예측이 의미 없음) → DEM 을 쓸 것.
- API: `POST /api/predict` 에 `color_scheme`(기본 `auto`), `reverse`(기본 `false`) 추가, 응답에
  `color_info`·`original_png_base64`. 선택지는 `GET /api/color-schemes`.

'등고선 데이터' 지원 포맷 (`terrain_io.py` `_PARSERS`):
`.npy` · `.csv` `.txt` `.dat` (숫자 격자) · `.asc` `.grd` (ESRI ASCII Grid, cellsize 자동 인식) ·
`.tif` `.tiff` (단일 밴드) · `.png` `.jpg` (회색조 높이맵).
→ 팀에서 쓰는 실제 포맷이 정해지면 `_PARSERS` 에 함수 하나만 추가.

## 강우 · 시점 조건

기본 체크포인트는 **`ai_model/checkpoints_p2_wet8/best.pt`** (젖은 픽셀 가중 L1, 128px/10.5M,
cond_dim=2). 화면의 두 슬라이더 값이 `rain_mm`, `time_s` 로 전달돼 예측이 실제로 달라진다.
학습 범위는 강우 20~80mm, 경과 시간 0~120초.

**경과 시간이 짧을수록 물이 많다.** 학습 데이터(OpenFOAM)는 비가 초반에 내리고 이후 빠지는
시뮬레이션이라, 정답의 평균 수심이 t=18s 0.039m -> t=120s 0.019m 로 단조 감소한다.
그래서 "이 지형 침수 예상"을 묻는 기본 조회 시점은 120초가 아니라 **18초**(`DEFAULT_TIME_S`)다.

결과 카드의 "시점 애니메이션 만들기"는 0~120초를 20초 간격 7프레임으로 예측해
물이 차올랐다 빠지는 과정을 재생한다 (프레임당 서버 호출 1회).

체크포인트의 학습 해상도와 조건 개수(cond_dim)는 가중치에서 자동 판별되므로,
구버전 체크포인트(`checkpoints_film`, cond_dim=1)로 되돌려도 그대로 뜬다 -
이 경우 시점 슬라이더는 자동으로 숨겨진다.

성능 근거는 `ai_model/IMPROVEMENT_GUIDE.md` 와 `ai_model/eval_p2_wet8/summary.md`.

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
