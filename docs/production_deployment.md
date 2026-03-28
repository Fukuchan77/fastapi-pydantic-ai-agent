# Production Deployment Guide

This guide covers production deployment considerations for the fastapi-pydantic-ai-agent application, including reverse proxy configuration, security hardening, and scalability patterns.

## Table of Contents

- [Reverse Proxy Configuration](#reverse-proxy-configuration)
  - [Request Size Limits](#request-size-limits)
  - [Nginx](#nginx)
  - [Apache](#apache)
  - [AWS Application Load Balancer (ALB)](#aws-application-load-balancer-alb)
  - [Cloudflare](#cloudflare)
- [Security Hardening](#security-hardening)
- [Scalability Patterns](#scalability-patterns)
- [Monitoring and Observability](#monitoring-and-observability)

---

## Reverse Proxy Configuration

### Request Size Limits

**⚠️ CRITICAL SECURITY NOTICE**

The application's [`RequestSizeLimitMiddleware`](../app/middleware/request_size.py) only validates the `Content-Length` header. It **CANNOT** enforce limits on:

- Requests using `Transfer-Encoding: chunked` (no Content-Length header)
- Requests that omit the `Content-Length` header entirely
- Malicious clients that lie about the Content-Length value

**This allows attackers to bypass the application-level size limit and cause denial-of-service attacks by sending unlimited request bodies.**

**SOLUTION**: Configure your reverse proxy or load balancer to enforce actual body size limits at the infrastructure layer.

---

### Nginx

Nginx is the recommended reverse proxy for production deployments due to its performance and robust body size validation.

#### Basic Configuration

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    # CRITICAL: Enforce maximum request body size
    # This applies to ALL encoding types, including chunked
    client_max_body_size 10M;

    # Optional: Return 413 immediately for large Content-Length
    # (faster than waiting for body upload)
    client_body_buffer_size 128K;

    # Optional: Timeout for client body upload
    client_body_timeout 30s;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running agent requests
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
    }
}
```

#### Advanced Configuration with Rate Limiting

```nginx
# Define rate limiting zone (outside server block)
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=60r/m;

server {
    listen 80;
    server_name api.yourdomain.com;

    # Body size limit
    client_max_body_size 10M;

    location /v1/ {
        # Apply rate limiting
        limit_req zone=api_limit burst=10 nodelay;
        limit_req_status 429;

        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Disable buffering for SSE streaming endpoints
        proxy_buffering off;
        proxy_cache off;

        # Timeouts
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
    }

    # Health check endpoint (no rate limit)
    location /health {
        proxy_pass http://localhost:8000;
        access_log off;
    }
}
```

#### Trusted Proxies Configuration

When running behind Nginx, configure the application to trust the proxy's IP address:

```bash
# .env or environment variables
TRUSTED_PROXIES=["127.0.0.1", "10.0.0.0/8"]  # Adjust to your Nginx server IPs
```

This ensures the application correctly extracts the real client IP from `X-Forwarded-For` headers for rate limiting.

---

### Apache

Apache HTTP Server can also be used as a reverse proxy with body size validation.

#### Basic Configuration

```apache
<VirtualHost *:80>
    ServerName api.yourdomain.com

    # CRITICAL: Enforce maximum request body size (in bytes)
    # 10MB = 10 * 1024 * 1024 = 10485760 bytes
    LimitRequestBody 10485760

    # Optional: Limit total request line + headers size
    LimitRequestLine 8190
    LimitRequestFieldSize 8190

    ProxyPreserveHost On
    ProxyPass / http://localhost:8000/
    ProxyPassReverse / http://localhost:8000/

    # Forward client IP
    RequestHeader set X-Forwarded-Proto "http"
    RequestHeader set X-Forwarded-For "%{REMOTE_ADDR}s"
</VirtualHost>
```

#### Advanced Configuration with mod_security

```apache
<VirtualHost *:80>
    ServerName api.yourdomain.com

    # Body size limit
    LimitRequestBody 10485760

    # Enable mod_security
    SecRuleEngine On

    # Additional security rules
    SecRule REQUEST_HEADERS:Content-Length "^[0-9]+$" "id:100,phase:1,deny,status:400,msg:'Invalid Content-Length header'"

    ProxyPreserveHost On
    ProxyPass / http://localhost:8000/
    ProxyPassReverse / http://localhost:8000/

    # Timeouts
    ProxyTimeout 60

    # Forward headers
    RequestHeader set X-Forwarded-Proto "http"
    RequestHeader set X-Forwarded-For "%{REMOTE_ADDR}s"
</VirtualHost>
```

---

### AWS Application Load Balancer (ALB)

AWS ALB automatically enforces a **maximum request body size of 1MB for HTTP/1.1** and **10MB for HTTP/2** (as of 2024). No additional configuration is required for basic protection.

#### ALB Target Group Configuration

```yaml
# CloudFormation or Terraform example
TargetGroup:
  Type: AWS::ElasticLoadBalancingV2::TargetGroup
  Properties:
    HealthCheckPath: /health
    HealthCheckIntervalSeconds: 30
    HealthCheckTimeoutSeconds: 5
    HealthyThresholdCount: 2
    UnhealthyThresholdCount: 3
    Port: 8000
    Protocol: HTTP
    VpcId: !Ref VPC
    TargetType: ip
    Matcher:
      HttpCode: 200
    # Deregistration delay for graceful shutdown
    DeregistrationDelayTimeoutSeconds: 30
```

#### Security Group Configuration

```yaml
SecurityGroup:
  Type: AWS::EC2::SecurityGroup
  Properties:
    GroupDescription: Application security group
    VpcId: !Ref VPC
    SecurityGroupIngress:
      # Allow traffic from ALB only
      - IpProtocol: tcp
        FromPort: 8000
        ToPort: 8000
        SourceSecurityGroupId: !Ref ALBSecurityGroup
```

#### Trusted Proxies for ALB

```bash
# .env - Trust ALB's VPC CIDR
TRUSTED_PROXIES=["10.0.0.0/8"]  # Adjust to your VPC CIDR
```

---

### Cloudflare

Cloudflare enforces request size limits automatically based on your plan:

- **Free Plan**: 100MB max upload
- **Pro/Business**: 100MB max upload
- **Enterprise**: 500MB max upload (configurable)

#### Cloudflare Configuration

1. **Enable "Under Attack Mode"** (optional) for DDoS protection
2. **Configure Page Rules** for API endpoints:

```
URL Pattern: api.yourdomain.com/v1/*
Settings:
  - Cache Level: Bypass
  - Security Level: High
  - Browser Integrity Check: On
```

3. **Rate Limiting Rule** (Business+ plan):

```
Rule Expression: (http.request.uri.path contains "/v1/")
Action: Block
Duration: 60 seconds
Requests per period: 100
```

#### Trusted Proxies for Cloudflare

Cloudflare's IP ranges change frequently. Use their [IP ranges API](https://www.cloudflare.com/ips/) or configure a wide range:

```bash
# .env - Trust Cloudflare IPv4 ranges (verify current ranges)
TRUSTED_PROXIES=["173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22", "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20", "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13", "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22"]
```

**Note**: Cloudflare sets `CF-Connecting-IP` header in addition to `X-Forwarded-For`. The application currently only reads `X-Forwarded-For`, which is sufficient for rate limiting.

---

## Security Hardening

### Environment Variables

**Never hardcode secrets in Docker images or code:**

```bash
# Use secrets management
docker run -d \
  -e API_KEY="$(aws secretsmanager get-secret-value --secret-id api-key --query SecretString --output text)" \
  -e LLM_API_KEY="$(aws secretsmanager get-secret-value --secret-id llm-key --query SecretString --output text)" \
  fastapi-pydantic-ai-agent:latest
```

### HTTPS Enforcement

**Always use HTTPS in production.** Configure your reverse proxy to:

1. Redirect HTTP to HTTPS
2. Enable HSTS headers
3. Use TLS 1.2+ only

```nginx
# Nginx example
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Strong SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers on;

    # HSTS header (already set by app, but reinforce)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # ... rest of configuration
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### API Key Rotation

Implement a key rotation strategy:

1. Use separate keys per environment (dev/staging/prod)
2. Rotate keys quarterly or after security incidents
3. Support multiple valid keys during transition periods

---

## Scalability Patterns

### Multi-Instance Deployment

For high availability, run multiple application instances behind a load balancer:

```bash
# Docker Compose example with multiple replicas
docker-compose up --scale api=3
```

**IMPORTANT**: When running multiple instances, **you must replace the in-memory stores with external services**:

#### Replace InMemorySessionStore

```python
# Example: Redis-backed SessionStore
from redis.asyncio import Redis
from pydantic_ai.messages import ModelMessage

class RedisSessionStore:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def get_history(self, session_id: str) -> list[ModelMessage]:
        data = await self.redis.get(f"session:{session_id}")
        if not
            return []
        return [ModelMessage.model_validate_json(msg) for msg in json.loads(data)]

    async def save_history(self, session_id: str, messages: list[ModelMessage]) -> None:
        data = json.dumps([msg.model_dump_json() for msg in messages])
        await self.redis.setex(f"session:{session_id}", 3600, data)  # 1 hour TTL

# In app/main.py lifespan
redis_client = Redis.from_url(os.environ["REDIS_URL"])
app.state.session_store = RedisSessionStore(redis_client)
```

#### Replace InMemoryVectorStore

```python
# Example: Chroma vector store
import chromadb
from app.stores.vector_store import VectorStore

class ChromaVectorStore:
    def __init__(self, collection_name: str):
        self.client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"])
        self.collection = self.client.get_or_create_collection(collection_name)

    async def add_documents(self, chunks: list[str]) -> None:
        ids = [str(uuid4()) for _ in chunks]
        self.collection.add(documents=chunks, ids=ids)

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["documents"][0] if results["documents"] else []

    async def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)

# In app/main.py lifespan
app.state.vector_store = ChromaVectorStore("fastapi-pydantic-ai-agent")
```

---

## Monitoring and Observability

### Pydantic Logfire

Configure Logfire in production:

```bash
# .env
LOGFIRE_TOKEN=your-logfire-token
LOGFIRE_SERVICE_NAME=fastapi-pydantic-ai-agent-prod
```

### Health Check Monitoring

Set up uptime monitoring on the `/health` and `/health/ready` endpoints:

```bash
# Example: Kubernetes
```
