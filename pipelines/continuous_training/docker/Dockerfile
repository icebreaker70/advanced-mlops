FROM python:3.11-slim
LABEL maintainer="otzslayer@gmail.com"

ARG USER_HOME=/home/codespace

# TODO: 인자 정보 추가
# 모든 경로는 프로젝트 루트 폴더를 기준으로 생각
ARG UTIL_PATH=utils
ARG PREPROCESSING_PATH=pipelines/continuous_training/data_preprocessing
ARG REQUIREMENTS_PATH=pipelines/continuous_training/docker
ARG TRAINING_PATH=pipelines/continuous_training/training

RUN groupadd --gid 1000 codespace \
    && useradd --uid 1000 --gid codespace --shell /bin/bash --create-home codespace

COPY --chown=codespace:codespace ${UTIL_PATH}/ ${USER_HOME}/utils
COPY --chown=codespace:codespace ${PREPROCESSING_PATH}/preprocessor.py \
    ${USER_HOME}/data_preprocessing/
COPY --chown=codespace:codespace ${TRAINING_PATH}/trainer.py \
    ${USER_HOME}/training/    
COPY --chown=codespace:codespace ${REQUIREMENTS_PATH}/requirements.txt \
    ${USER_HOME}/

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

USER codespace

RUN mkdir -p ${USER_HOME}/artifacts \
    && pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r ${USER_HOME}/requirements.txt

WORKDIR ${USER_HOME}
