FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    tini \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# UID pinned to 1000 to match HostConfig.User.
RUN useradd -m -u 1000 -s /bin/bash swe-user

# Secrets directory created with restrictive perms; the host injects
# files into this directory before container start.
RUN mkdir -p /tmp/open-swe/secrets \
    && chown -R swe-user:swe-user /tmp/open-swe \
    && chmod 700 /tmp/open-swe/secrets

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER swe-user
WORKDIR /workspace

ENV PATH=/home/swe-user/.local/bin:$PATH

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["sh", "-c", "test -f /tmp/open-swe/ready"]

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
