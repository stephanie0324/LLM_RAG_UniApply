# syntax=docker/dockerfile:1
FROM python:3.9.18-bullseye

# Update and install required system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libmagic-dev \
    ghostscript \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    ca-certificates && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN --mount=type=bind,source=./app/install/requirements.txt,target=/app/src/requirements.txt pip install --trusted-host pypi.org --trusted-host pypi.python.org -r /app/src/requirements.txt

# Create necessary directories with permissions
RUN mkdir -p /.cache && chmod -R 777 /.cache && mkdir -p /tmp && chmod -R 777 /tmp && mkdir -p /.paddleocr && chmod -R 777 /.paddleocr

# Set working directory
WORKDIR /app
COPY --chmod="777" /app /app

# Expose the port for streamlit
EXPOSE 7860

# Run the application with streamlit
ENTRYPOINT ["streamlit"]
CMD ["run", "web.py", "--server.port", "7860"]
