import os
from zoneinfo import ZoneInfo

# Cấu hình địa chỉ FastAPI và thời gian chờ request
API_BASE_URL = os.getenv("SNAKEGUARD_API_BASE_URL", "http://127.0.0.1:8000",).rstrip("/")
API_TIMEOUT = float(os.getenv("SNAKEGUARD_API_TIMEOUT", "120"))
VN_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
