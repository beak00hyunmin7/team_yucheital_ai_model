"""run_server.bat 용: 같은 공유기의 다른 PC 가 접속할 주소를 출력한다."""
import socket

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("192.168.255.255", 9))   # 실제로 보내지 않고, 나가는 쪽 랜 주소만 알아낸다
    ip = s.getsockname()[0]
    s.close()
    print(f"      teammates  : http://{ip}:8000")
except OSError:
    print("      teammates  : run ipconfig and use http://[IPv4 address]:8000")
