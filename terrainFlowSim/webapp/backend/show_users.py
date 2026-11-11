"""yuche_ai_db 의 가입 계정과 사용자별 분석 기록 수를 보여준다 (비밀번호 해시는 출력하지 않음)."""
import pymysql

c = pymysql.connect(host="127.0.0.1", user="root", password="1234", database="yuche_ai_db", charset="utf8mb4")
cur = c.cursor()
cur.execute("""
    SELECT u.id, u.email, u.created_at, COUNT(h.id)
    FROM users u LEFT JOIN analysis_history h ON h.user_email = u.email
    GROUP BY u.id, u.email, u.created_at ORDER BY u.id
""")
rows = cur.fetchall()
print(f"\n  yuche_ai_db.users : {len(rows)} accounts\n")
print(f"  {'id':>4}  {'email':<36} {'joined':<20} records")
print("  " + "-" * 72)
for i, email, created, n in rows:
    print(f"  {i:>4}  {email:<36} {str(created):<20} {n}")
print()
