"""yuche_ai_db 의 모든 테이블을 통째로 보여주고 CSV 로도 저장한다.

- 화면: 테이블마다 전체 행 (긴 값은 잘라서 표시)
- 파일: webapp/data/db_export/<테이블>.csv (엑셀로 열면 전체 값이 다 보임)
"""
import csv
import datetime
from pathlib import Path

import pymysql

DB = "yuche_ai_db"
OUT = Path(__file__).resolve().parent.parent / "data" / "db_export"
OUT.mkdir(parents=True, exist_ok=True)
MAXW = 40   # 화면에서 한 칸 최대 글자 수

c = pymysql.connect(host="127.0.0.1", user="root", password="1234", database=DB, charset="utf8mb4")
cur = c.cursor()
cur.execute("SHOW TABLES")
tables = [t[0] for t in cur.fetchall()]
print(f"\n=== {DB} : 테이블 {len(tables)}개 ({', '.join(tables)}) ===")

for t in tables:
    cur.execute(f"SELECT * FROM `{t}`")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    with open(OUT / f"{t}.csv", "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀 한글 깨짐 방지
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    def cell(v):
        s = "" if v is None else str(v)
        return s if len(s) <= MAXW else s[: MAXW - 3] + "..."

    table = [[cell(v) for v in r] for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in table]) for i, h in enumerate(cols)]
    print(f"\n--- {t} : {len(rows)}행 ---")
    print("  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(cols)))
    print("  " + "-+-".join("-" * w_ for w_ in widths))
    for r in table:
        print("  " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(r)))

print(f"\nCSV 저장 위치: {OUT}")
print(f"(저장 시각 {datetime.datetime.now():%Y-%m-%d %H:%M:%S})\n")
