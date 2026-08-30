# MediaPipe 는 아직 3.13 휠을 제공하지 않는다. 3.12 로 고정한다.
FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless 도 libglib 는 필요로 한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 의존성을 먼저 복사해 레이어 캐시를 살린다. 소스만 바뀌면 재설치하지 않는다.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# models/*.task 는 저장소에 포함돼 있어 런타임 다운로드가 없다.
COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host ${HOST} --port ${PORT}"]
