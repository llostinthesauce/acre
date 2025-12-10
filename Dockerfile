FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libsndfile1 \
    python3-tk \
    tk \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

COPY vendor ./vendor

RUN if [ -d vendor ] && [ "$(ls -A vendor)" ]; then \
        pip install --no-index --find-links=vendor -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY acre_app/ ./acre_app/
COPY model_manager/ ./model_manager/
COPY platform_utils.py .
COPY transparent-logo.png .

RUN mkdir -p config

RUN mkdir -p models history outputs vendor

ENV PYTHONPATH=/app
ENV DISPLAY=:99

COPY app.py .

RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
exec "$@"\n\
' > /app/start-xvfb.sh && chmod +x /app/start-xvfb.sh

CMD ["/app/start-xvfb.sh", "python", "app.py"]
