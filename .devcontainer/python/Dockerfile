FROM python:3.11

WORKDIR /tmp/work

COPY requirements.txt ${PWD}

RUN pip install -U pip && \
    pip install -r requirements.txt
