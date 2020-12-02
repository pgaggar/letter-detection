FROM python:3.7-slim

ENV PATH="/opt/program:${PATH}"

RUN apt-get update && apt-get install -y python-opencv

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY . /opt/program

WORKDIR /opt/program