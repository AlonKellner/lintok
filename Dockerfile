ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/base:debian
FROM $BASE_IMAGE AS base
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx
ENV PATH="/root/.local/bin/:$PATH"
RUN uv self update

WORKDIR /app

RUN --mount=type=cache,dst=/root/.cache/ \
    echo general-clean && rm -rf /opt/conda && rm -rf /var/lib/apt/lists/* && apt clean && \
    echo apt-setup && apt update && apt upgrade -y && \
    echo apt-tools && apt install -y --no-install-recommends bash ca-certificates curl file git \
    inotify-tools jq libgl1 lsof vim nano tmux nginx openssh-server procps pkg-config cmake \
    rsync sudo software-properties-common unzip wget zip && apt autoremove -y && apt update && apt upgrade -y

COPY dev-pyproject/ ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv python install --preview --default

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,dst=/root/.cache/ \
    uv sync

ARG WORKDIR=/app
WORKDIR ${WORKDIR}

FROM base AS python
ENTRYPOINT [ "/app/.venv/bin/python" ]

FROM base AS bare
ENTRYPOINT []
