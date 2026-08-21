FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    datasets \
    seqeval \
    scikit-learn \
    pandas \
    numpy \
    tqdm \
    accelerate \
    tensorboard \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    psycopg2-binary \
    neo4j \
    redis \
    python-multipart \
    jupyterlab \
    ipywidgets

EXPOSE 8000 8888
CMD ["/bin/bash"]
