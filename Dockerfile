FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS example_algorithm_amd64

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r user && \
    useradd -m --no-log-init -r -g user user

USER user
WORKDIR /opt/app

# Inherit PyTorch and CUDA packages from the base image
RUN python -m venv \
    --system-site-packages \
    --without-pip \
    /home/user/venv

ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# nnU-Net source and the inference-compatible LadleResEncL trainer class.
COPY --chown=user:user vendor/nnunetv2 /opt/app/nnunetv2

# Ensure the vendored nnU-Net is imported before any installed package.
ENV PYTHONPATH="/opt/app"

COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user CONTAINER_VERSION.txt /opt/app/

LABEL org.grand-challenge.api-method="invoke"

EXPOSE 4743

ENTRYPOINT ["python", "app.py"]
