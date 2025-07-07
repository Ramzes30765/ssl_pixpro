FROM python:3.10.16-bookworm

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app

RUN curl -sSL https://install.python-poetry.org | python3.10 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

RUN pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124