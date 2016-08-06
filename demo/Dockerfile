FROM python:3.5-slim
MAINTAINER Binh Le <lebinh.it@gmail.com>

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=true \
    KERAS_BACKEND=tensorflow

WORKDIR /opt/demo

CMD ["gunicorn", "--bind=0.0.0.0:80", "app:__hug_wsgi__"]

COPY . .
