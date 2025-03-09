# syntax=docker/dockerfile:1
FROM python:3.9.18-bullseye

RUN apt update && apt install -y libgl1-mesa-dev libmagic-dev ghostscript poppler-utils tesseract-ocr libreoffice && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=./app/install/requirements.txt,target=/app/src/requirements.txt pip install -r /app/src/requirements.txt

# RUN groupadd -r llm && useradd --no-log-init -r -g llm llm
RUN  mkdir -p /.cache && chmod -R 777 /.cache && mkdir -p /tmp && chmod -R 777 /tmp && mkdir -p /.paddleocr && chmod -R 777 /.paddleocr

RUN mkdir -p /app && chmod -R 777 /app
# USER llm:llm
WORKDIR /app
COPY --chmod="777" . /app

EXPOSE 7860

ENTRYPOINT ["streamlit"]
CMD ["run", "web.py", "--server.port", "7860"]
