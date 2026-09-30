# FROM python:3.12-slim

# WORKDIR /app

# # System deps Playwright's Chromium needs at runtime.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     wget gnupg ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# RUN python -m playwright install --with-deps chromium

# COPY app ./app

# RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
# USER appuser

# EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]

FROM python:3.12-slim 

WORKDIR /app 

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# 👇 VERY IMPORTANT
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# install browsers to shared location
RUN python -m playwright install --with-deps chromium

COPY app ./app 

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /ms-playwright \
    && chown -R appuser:appuser /app /ms-playwright

USER appuser 

EXPOSE 8000 

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]