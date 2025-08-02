# Monitoring and Observability

Comprehensive monitoring setup for the Analog PDE Solver framework.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards  
- **Health Checks**: Application health monitoring
- **Alertmanager**: Alert routing and notification
- **Log Aggregation**: Centralized logging

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   App       │───▶│ Prometheus  │───▶│  Grafana    │
│ (metrics)   │    │             │    │ (dashboards)│
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐    ┌─────────────┐
                   │Alertmanager │───▶│ Slack/Email │
                   │             │    │ (notifications)
                   └─────────────┘    └─────────────┘
```

## Metrics

### Application Metrics

The analog PDE solver exposes the following metrics:

#### Core Solver Metrics
- `analog_pde_solver_solve_duration_seconds`: Time taken to solve PDE
- `analog_pde_solver_convergence_iterations`: Number of iterations to converge
- `analog_pde_solver_convergence_failures_total`: Total convergence failures
- `analog_pde_solver_crossbar_utilization`: Crossbar array utilization ratio
- `analog_pde_solver_noise_level`: Current analog noise level

#### Performance Metrics
- `analog_pde_solver_memory_usage_bytes`: Current memory usage
- `analog_pde_solver_cpu_usage_percent`: CPU utilization
- `analog_pde_solver_active_simulations`: Number of active simulations

#### SPICE Simulator Metrics
- `spice_simulations_total`: Total SPICE simulations run
- `spice_simulation_duration_seconds`: SPICE simulation duration
- `spice_simulation_failures_total`: Failed SPICE simulations
- `spice_simulation_timeouts_total`: Timed out SPICE simulations

#### Hardware Testing Metrics
- `hardware_test_duration_seconds`: Hardware test execution time
- `hardware_test_failures_total`: Hardware test failures
- `verilog_synthesis_duration_seconds`: Verilog synthesis time

### System Metrics

Standard system metrics via node_exporter:

- CPU usage, load average
- Memory usage and availability  
- Disk usage and I/O
- Network statistics

## Health Checks

### Endpoints

- **`/health`**: Basic health check (HTTP 200 if healthy)
- **`/health/detailed`**: Comprehensive health information
- **`/health/ready`**: Readiness probe for Kubernetes
- **`/health/live`**: Liveness probe for Kubernetes

### Health Check Components

1. **Application Health**
   - Package import test
   - Version information
   - Uptime tracking

2. **System Resources**
   - Memory availability
   - Disk space
   - CPU load

3. **External Dependencies**
   - NgSpice availability
   - Verilog tools (iverilog, verilator)
   - Python package dependencies

4. **File Permissions**
   - Temp directory access
   - Log file write permissions
   - Data directory permissions

### Running Health Checks

```bash
# Simple health check
python monitoring/health-checks/health_check.py --simple

# Comprehensive health check
python monitoring/health-checks/health_check.py

# Via HTTP endpoint (if server running)
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
```

## Alerts

### Alert Categories

#### Critical Alerts (Page immediately)
- Application down (`AnalogPDESolverDown`)
- Out of memory (`OutOfMemory`)
- Multiple convergence failures (`ConvergenceFailure`)

#### Warning Alerts (Notify during business hours)
- High error rate (`HighErrorRate`)
- High response time (`HighResponseTime`)
- High resource usage (`HighMemoryUsage`, `HighCPUUsage`)
- Slow convergence (`SlowConvergence`)

#### Informational Alerts (Log only)
- Crossbar utilization trends
- Performance degradation patterns

### Alert Configuration

Alerts are defined in `monitoring/prometheus/rules/analog_pde_solver.yml`.

#### Customizing Thresholds

```yaml
# Example: Adjust memory usage threshold
- alert: HighMemoryUsage
  expr: (process_resident_memory_bytes{job="analog-pde-solver"} / 1024 / 1024 / 1024) > 8  # 8GB instead of 4GB
  for: 10m
```

#### Alert Routing

Configure alert routing in Alertmanager:

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@company.com'
        subject: 'Warning: {{ .GroupLabels.alertname }}'
        body: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## Dashboards

### Available Dashboards

1. **Overview Dashboard** (`analog-pde-solver-overview.json`)
   - System health status
   - Request rate and response time
   - PDE solver performance
   - Resource utilization

2. **Performance Dashboard**
   - Convergence metrics
   - Iteration counts
   - Solve duration trends
   - Memory usage patterns

3. **Hardware Dashboard**
   - SPICE simulation metrics
   - Verilog synthesis status
   - Crossbar utilization
   - Hardware test results

4. **System Dashboard**
   - CPU, memory, disk usage
   - Network statistics
   - Process information

### Importing Dashboards

```bash
# Import via Grafana UI
# Settings → Data Sources → Add Prometheus data source
# Dashboards → Import → Upload JSON file

# Or via API
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/analog-pde-solver-overview.json
```

### Creating Custom Dashboards

1. **Navigate to Grafana** (http://localhost:3000)
2. **Add Panel** → Select visualization type
3. **Configure Query** using PromQL
4. **Set Display Options** (titles, axes, thresholds)
5. **Save Dashboard**

Example queries:

```promql
# Average solve time over 5 minutes
rate(analog_pde_solver_solve_duration_seconds_sum[5m]) / 
rate(analog_pde_solver_solve_duration_seconds_count[5m])

# Success rate
rate(analog_pde_solver_solve_success_total[5m]) / 
rate(analog_pde_solver_solve_attempts_total[5m]) * 100

# 95th percentile response time
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket{job="analog-pde-solver"}[5m])
)
```

## Docker Deployment

### Quick Start

```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
```

### Production Deployment

```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules/:/etc/prometheus/rules/:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro

volumes:
  prometheus-data:
  grafana-data:
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**
   - Check Prometheus targets: http://localhost:9090/targets
   - Verify application is exposing metrics on correct port
   - Check firewall/network connectivity

2. **Alerts not firing**
   - Verify alert rules syntax in Prometheus UI
   - Check Alertmanager configuration
   - Test notification channels

3. **Dashboard not loading**
   - Verify Prometheus data source configuration
   - Check dashboard JSON for syntax errors
   - Ensure Grafana has access to Prometheus

4. **High memory usage**
   - Adjust Prometheus retention settings
   - Reduce scrape frequency for less critical metrics
   - Use recording rules for expensive queries

### Debug Commands

```bash
# Check Prometheus configuration
promtool check config monitoring/prometheus/prometheus.yml

# Validate alert rules
promtool check rules monitoring/prometheus/rules/*.yml

# Test Grafana dashboard
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/dashboards/uid/DASHBOARD_UID

# Health check application
curl http://localhost:8000/health/detailed | jq .
```

### Performance Tuning

1. **Prometheus**
   - Adjust `scrape_interval` based on needs
   - Use `recording rules` for complex queries
   - Configure appropriate `retention.time`

2. **Grafana**
   - Use `query caching` for expensive dashboards
   - Set appropriate `refresh intervals`
   - Limit `time ranges` for queries

3. **Application**
   - Batch metric updates when possible
   - Use `histogram` metrics for timing data
   - Avoid high-cardinality labels

## Best Practices

### Metric Design
- Use consistent naming conventions
- Include units in metric names
- Keep cardinality reasonable
- Use histograms for timing metrics

### Alert Design
- Alert on symptoms, not causes
- Use appropriate severity levels
- Include actionable information in descriptions
- Avoid alert fatigue with proper thresholds

### Dashboard Design
- Group related metrics logically
- Use consistent time ranges
- Include context and annotations
- Test with real data scenarios