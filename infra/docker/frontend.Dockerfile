FROM python:3.12-slim

WORKDIR /app

# Cài dependency riêng cho frontend để image không chứa các thư viện AI nặng
COPY frontend/requirements.txt ./frontend-requirements.txt
RUN pip install --no-cache-dir -r frontend-requirements.txt

# Chỉ copy mã nguồn Streamlit vào frontend image
COPY frontend/streamlit_app/ .

# Copy cấu hình Streamlit từ root project
COPY .streamlit/ /app/.streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
