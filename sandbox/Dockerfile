FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils libheif1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY bridge_client.py /opt/sandbox/bridge_client.py
COPY executor.py /opt/sandbox/executor.py
COPY docs/ /opt/sandbox/docs/

RUN mkdir -p /work && chmod 777 /work

WORKDIR /work

ENTRYPOINT ["python", "/opt/sandbox/executor.py"]
