# 404-GEN Miner

Minimal Dockerized miner service for validator audit.

Build from the repository root:

```bash
docker build -f docker/Dockerfile .
```

The container exposes the batch generation API on port `10006`:

```text
GET  /health
GET  /status
POST /generate
GET  /results
```

No credentials are stored in this repository. Runtime secrets and model access
must be provided through the deployment environment.
