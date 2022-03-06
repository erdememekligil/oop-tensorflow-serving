FROM tensorflow/tensorflow:2.8.0-gpu

LABEL maintainer="Erdem Emekligil erdememekligil@gmail.com"

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt
