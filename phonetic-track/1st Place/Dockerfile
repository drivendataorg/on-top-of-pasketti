# Mirrors the DrivenData Pasketti Phonetic runtime image used to evaluate
# submissions. We only need this image for local end-to-end submission tests
# — the platform itself rebuilds the environment from your submission.zip.
FROM nvcr.io/nvidia/pytorch:24.07-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NEMO_CACHE_DIR=/opt/nemo_cache

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r /workspace/requirements.txt

CMD ["/bin/bash"]
