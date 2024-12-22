FROM python:3.10.6-slim
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6

# FROM tensorflow/tensorflow:2.10.0

COPY requirements.txt /requirements.txt
# COPY requirements_prod.txt /requirements_prod.txt

RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY wine_advisor /wine_advisor
# RUN pip install wine_advisor

CMD uvicorn wine_advisor.api.api_wine:app --host 0.0.0.0 --port $PORT
