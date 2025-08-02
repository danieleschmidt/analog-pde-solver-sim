# Deployment Guide

This guide covers deployment options for the Analog PDE Solver framework.

## Container-based Deployment

The project uses Docker for consistent deployment across environments.

### Available Images

- **Production** (`analog-pde-solver:latest`): Minimal production image
- **Development** (`analog-pde-solver:dev`): Full development environment
- **Hardware** (`analog-pde-solver:hardware`): Includes SPICE and Verilog tools

### Quick Start

```bash
# Development environment
docker-compose up -d dev

# Production deployment
docker-compose up -d prod

# Testing environment
docker-compose run --rm test

# Documentation server
docker-compose up -d docs
```

### Build Process

The multi-stage Dockerfile optimizes for:
- **Security**: Non-root user, minimal attack surface
- **Size**: Production image ~200MB vs development ~1.2GB  
- **Performance**: Cached layers for faster rebuilds
- **Flexibility**: Multiple targets for different use cases

## Production Deployment

### Docker Swarm

```yaml
# docker-stack.yml
version: '3.8'

services:
  analog-pde-solver:
    image: analog-pde-solver:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    networks:
      - analog-pde-net
    secrets:
      - api_keys
    environment:
      - LOG_LEVEL=INFO
      - DEBUG_MODE=false

networks:
  analog-pde-net:
    driver: overlay

secrets:
  api_keys:
    external: true
```

Deploy with:
```bash
docker stack deploy -c docker-stack.yml analog-pde
```

### Kubernetes

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analog-pde-solver
  labels:
    app: analog-pde-solver
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analog-pde-solver
  template:
    metadata:
      labels:
        app: analog-pde-solver
    spec:
      containers:
      - name: analog-pde-solver
        image: analog-pde-solver:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: DEBUG_MODE
          value: "false"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: analog-pde-solver-service
spec:
  selector:
    app: analog-pde-solver
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy with:
```bash
kubectl apply -f k8s-deployment.yml
```

## Cloud Deployment

### AWS ECS

1. **Create ECR repository:**
```bash
aws ecr create-repository --repository-name analog-pde-solver
```

2. **Build and push image:**
```bash
# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com

# Build and tag
docker build -t analog-pde-solver:latest .
docker tag analog-pde-solver:latest <account>.dkr.ecr.us-west-2.amazonaws.com/analog-pde-solver:latest

# Push
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/analog-pde-solver:latest
```

3. **Create ECS task definition and service**

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/analog-pde-solver

# Deploy to Cloud Run
gcloud run deploy analog-pde-solver \
  --image gcr.io/PROJECT_ID/analog-pde-solver \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure Container Instances

```bash
# Create resource group
az group create --name analog-pde-rg --location eastus

# Create ACR
az acr create --resource-group analog-pde-rg --name analogpderegistry --sku Basic

# Build and push
az acr build --registry analogpderegistry --image analog-pde-solver:latest .

# Deploy container instance
az container create \
  --resource-group analog-pde-rg \
  --name analog-pde-solver \
  --image analogpderegistry.azurecr.io/analog-pde-solver:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000
```

## Hardware-specific Deployment

For deployments requiring SPICE or Verilog tools:

### SPICE Simulation Cluster

```yaml
# docker-compose.spice.yml
version: '3.8'

services:
  spice-worker:
    image: analog-pde-solver:hardware
    deploy:
      replicas: 4
    volumes:
      - spice-models:/usr/share/ngspice/models/custom
      - simulation-results:/workspace/results
    environment:
      - SPICE_PARALLEL_JOBS=4
      - WORKER_MODE=spice
    command: python -m analog_pde_solver.workers.spice_worker

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

volumes:
  spice-models:
  simulation-results:
  redis-data:
```

### GPU-accelerated Deployment

For GPU acceleration with CUDA:

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# ... base setup ...

# Install GPU-specific packages
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Set GPU environment
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Deploy with GPU support:
```bash
docker run --gpus all analog-pde-solver:gpu
```

## Monitoring and Logging

### Prometheus Metrics

Add to docker-compose.yml:
```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
```

### Centralized Logging

```yaml
  fluentd:
    image: fluent/fluentd:v1.14-debian
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
      - /var/log:/var/log:ro
    ports:
      - "24224:24224"
    environment:
      FLUENTD_CONF: fluent.conf
```

## Security Considerations

### Image Security

1. **Scan images regularly:**
```bash
docker scan analog-pde-solver:latest
```

2. **Use distroless base images for production:**
```dockerfile
FROM gcr.io/distroless/python3
COPY --from=builder /app /app
USER 1000
```

3. **Sign images:**
```bash
docker trust sign analog-pde-solver:latest
```

### Runtime Security

1. **Use read-only filesystems:**
```yaml
services:
  app:
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
```

2. **Drop capabilities:**
```yaml
services:
  app:
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

3. **Use secrets management:**
```yaml
services:
  app:
    secrets:
      - api_key
      - database_password
```

## Performance Optimization

### Resource Limits

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Caching Strategy

1. **Build cache optimization:**
```bash
# Use BuildKit for advanced caching
DOCKER_BUILDKIT=1 docker build --cache-from analog-pde-solver:cache .
```

2. **Multi-stage optimization:**
```dockerfile
# Cache dependencies separately
FROM python:3.11-slim as deps
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM deps as app
COPY . .
```

## Troubleshooting

### Common Issues

1. **Out of memory errors:**
   - Increase container memory limits
   - Use memory profiling tools
   - Optimize algorithms for memory efficiency

2. **SPICE simulation failures:**
   - Verify NgSpice installation
   - Check model file paths
   - Enable debug logging

3. **Container startup failures:**
   - Check logs: `docker logs <container_id>`
   - Verify environment variables
   - Test health check endpoints

### Debug Mode

Enable debug mode for troubleshooting:
```bash
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
```

### Log Analysis

```bash
# Follow logs in real-time
docker-compose logs -f app

# Export logs for analysis
docker-compose logs app > app.log
```