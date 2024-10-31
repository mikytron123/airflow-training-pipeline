FROM apache/airflow:slim-2.10.2-python3.11
USER root
ARG AIRFLOW_VERSION=2.10.2
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

ENV MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update -y && apt-get install -y git

ENV GIT_PYTHON_REFRESH=quiet

COPY . .
USER airflow
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt



