FROM mcr.microsoft.com/devcontainers/base:debian AS default
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    chmod +x $HOME/.local/bin/uv $HOME/.local/bin/uvx && \
    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b $HOME/.local/bin v0.65.0 && \
    chmod +x $HOME/.local/bin/trivy
ENV PATH="/home/vscode/.local/bin/:/github/home/.local/bin:$PATH"
RUN uv self update && uvx --version && trivy --version

USER root

COPY dev-pyproject/ ./
RUN DEBIAN_FRONTEND=noninteractive mkdir -p /home/vscode/.cache && chmod 777 /home/vscode/.cache && \
    echo apt-tools && apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends apt-transport-https \
    ca-certificates gnupg curl pkg-config cmake libssl-dev git expect && \
    echo taplo && curl -fsSL https://github.com/tamasfe/taplo/releases/latest/download/taplo-linux-$(uname -m).gz \
    | gzip -d - | install -m 755 /dev/stdin /usr/local/bin/taplo && taplo --version && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    echo nodejs && apt-get install -y nodejs && npm install -g npm@latest && node -v && npm -v && \
    echo claude && npm install -g @anthropic-ai/claude-code && claude --version

COPY ./.devcontainer/migrate_claude.exp ./
USER vscode
RUN expect migrate_claude.exp
