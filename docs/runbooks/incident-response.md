# Incident Response Runbook

This runbook provides step-by-step procedures for responding to common incidents in the Analog PDE Solver system.

## General Incident Response Process

### 1. Initial Response (First 5 minutes)
- [ ] Acknowledge the alert
- [ ] Assess severity (Critical/High/Medium/Low)
- [ ] Check system status dashboards  
- [ ] Determine if immediate action is required
- [ ] Notify team if Critical/High severity

### 2. Investigation (Next 15 minutes)
- [ ] Check monitoring dashboards for patterns
- [ ] Review recent deployments/changes
- [ ] Examine application logs
- [ ] Run health checks
- [ ] Identify potential root cause

### 3. Mitigation (Immediate action)
- [ ] Implement temporary fix if available
- [ ] Scale resources if needed
- [ ] Rollback recent changes if suspected cause
- [ ] Document actions taken

### 4. Resolution and Follow-up
- [ ] Implement permanent fix
- [ ] Verify system stability
- [ ] Update documentation
- [ ] Conduct post-incident review

---

## Specific Incident Procedures

### Application Down (`AnalogPDESolverDown`)

**Symptoms**: HTTP health check failures, no response from application

#### Immediate Actions
1. **Check service status**:
   ```bash
   # Docker/Kubernetes
   docker ps | grep analog-pde-solver
   kubectl get pods -l app=analog-pde-solver
   
   # System service
   systemctl status analog-pde-solver
   ```

2. **Check application logs**:
   ```bash
   # Docker logs
   docker logs analog-pde-solver-container
   
   # System logs
   journalctl -u analog-pde-solver -f --lines=100
   
   # Application logs
   tail -f /var/log/analog-pde-solver/app.log
   ```

3. **Check system resources**:
   ```bash
   # Memory and CPU
   free -h
   top -p $(pgrep -f analog-pde-solver)
   
   # Disk space
   df -h
   
   # Network connectivity
   netstat -tlnp | grep :8000
   ```

#### Recovery Steps
1. **Restart the service**:
   ```bash
   # Docker
   docker restart analog-pde-solver-container
   
   # Kubernetes
   kubectl rollout restart deployment/analog-pde-solver
   
   # System service
   systemctl restart analog-pde-solver
   ```

2. **If restart fails, check configuration**:
   ```bash
   # Validate configuration
   python -c "from analog_pde_solver import config; config.validate()"
   
   # Check environment variables
   env | grep ANALOG_PDE
   
   # Test basic functionality
   python -c "import analog_pde_solver; print('Import successful')"
   ```

3. **Scale up if resource constrained**:
   ```bash
   # Kubernetes
   kubectl scale deployment analog-pde-solver --replicas=3
   
   # Docker Compose
   docker-compose up -d --scale analog-pde-solver=3
   ```

---

### High Memory Usage (`HighMemoryUsage`)

**Symptoms**: Memory usage > 4GB, potential OOM kills

#### Investigation
1. **Check memory consumption patterns**:
   ```bash
   # Process memory
   ps aux --sort=-%mem | head -10
   
   # Memory by process tree
   pstree -p $(pgrep -f analog-pde-solver) | head -20
   
   # Memory maps
   pmap -d $(pgrep -f analog-pde-solver)
   ```

2. **Check for memory leaks**:
   ```bash
   # Python memory profiling
   python -m memory_profiler analog_pde_solver/main.py
   
   # Monitor memory over time
   watch -n 5 'ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head'
   ```

#### Mitigation
1. **Immediate relief**:
   - Restart the service (frees leaked memory)
   - Reduce problem sizes if possible
   - Scale horizontally instead of vertically

2. **Temporary tuning**:
   ```bash
   # Increase memory limits (Kubernetes)
   kubectl patch deployment analog-pde-solver -p '{"spec":{"template":{"spec":{"containers":[{"name":"analog-pde-solver","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   
   # Set Python memory optimization
   export PYTHONOPTIMIZE=1
   export MALLOC_TRIM_THRESHOLD_=100000
   ```

3. **Long-term fixes**:
   - Implement memory-efficient algorithms
   - Add memory usage monitoring to code
   - Tune garbage collection settings

---

### SPICE Simulation Timeouts (`SPICESimulationTimeout`)

**Symptoms**: SPICE simulations not completing, timeout errors

#### Investigation
1. **Check SPICE process status**:
   ```bash
   # Running SPICE processes
   ps aux | grep ngspice
   
   # Check for zombie processes
   ps aux | grep -E "(ngspice|<defunct>)"
   
   # Check SPICE temporary files
   ls -la /tmp/*spice* /tmp/*ngspice*
   ```

2. **Examine SPICE logs**:
   ```bash
   # SPICE output logs
   tail -f temp/spice/*.log
   
   # SPICE error files
   find temp/spice -name "*.err" -exec cat {} \;
   ```

3. **Check system resources**:
   ```bash
   # CPU usage by SPICE
   top -p $(pgrep ngspice)
   
   # I/O wait
   iostat -x 1 5
   ```

#### Resolution
1. **Kill stuck SPICE processes**:
   ```bash
   # Kill all SPICE processes
   pkill -9 ngspice
   
   # Clean up temporary files
   rm -f /tmp/*spice* /tmp/*ngspice*
   find temp/spice -name "*.raw" -mtime +1 -delete
   ```

2. **Adjust SPICE configuration**:
   ```bash
   # Reduce simulation accuracy for speed
   export SPICE_OPTS=".options reltol=1e-3 abstol=1e-10 gmin=1e-10"
   
   # Set simulation timeout
   export MAX_SPICE_RUNTIME=120  # 2 minutes
   ```

3. **Restart SPICE service**:
   ```bash
   # Docker Compose
   docker-compose restart spice-sim
   
   # Check SPICE installation
   ngspice --version
   which ngspice
   ```

---

### Convergence Failures (`ConvergenceFailure`)

**Symptoms**: PDE solver unable to converge, repeated failures

#### Investigation
1. **Check problem characteristics**:
   ```bash
   # Review recent problem inputs
   ls -la temp/problems/
   
   # Check problem size and complexity
   python -c "
   import numpy as np
   problem_data = np.load('temp/problems/latest.npy')
   print(f'Problem size: {problem_data.shape}')
   print(f'Condition number: {np.linalg.cond(problem_data)}')
   "
   ```

2. **Examine solver parameters**:
   ```bash
   # Check solver configuration
   grep -r "tolerance\|max_iter" analog_pde_solver/config/
   
   # Review solver logs
   grep -i "convergence\|iteration" logs/solver.log | tail -50
   ```

#### Resolution
1. **Adjust solver parameters**:
   ```python
   # Relax convergence tolerance
   solver.set_tolerance(1e-4)  # from 1e-6
   
   # Increase maximum iterations
   solver.set_max_iterations(2000)  # from 1000
   
   # Try different preconditioner
   solver.set_preconditioner('jacobi')  # from 'none'
   ```

2. **Problem preprocessing**:
   ```python
   # Check matrix conditioning
   cond_num = np.linalg.cond(matrix)
   if cond_num > 1e12:
       # Apply regularization
       matrix += 1e-12 * np.eye(matrix.shape[0])
   
   # Scale the problem
   matrix_norm = np.linalg.norm(matrix, 'fro')
   matrix /= matrix_norm
   ```

3. **Switch to robust solver**:
   ```python
   # Use more stable solver for difficult problems
   solver = RobustAnalogSolver(
       method='gmres',
       preconditioner='ilu',
       restart=50
   )
   ```

---

### High Error Rate (`HighErrorRate`)

**Symptoms**: HTTP 5xx error rate > 10%, application errors

#### Investigation
1. **Check error patterns**:
   ```bash
   # Application error logs
   grep -E "(ERROR|CRITICAL)" logs/app.log | tail -50
   
   # HTTP error patterns
   grep " 50[0-9] " logs/access.log | tail -20
   
   # Exception stack traces
   grep -A 10 "Traceback" logs/app.log | tail -50
   ```

2. **Check external dependencies**:
   ```bash
   # Test SPICE connectivity
   timeout 5 ngspice --version
   
   # Check file system
   touch temp/test_write && rm temp/test_write
   
   # Network connectivity
   curl -I http://localhost:8000/health
   ```

#### Resolution
1. **Address common causes**:
   ```bash
   # Restart if configuration issues
   docker-compose restart analog-pde-solver
   
   # Clear temporary files if disk full
   find temp/ -type f -mtime +1 -delete
   
   # Check and fix permissions
   chmod -R 755 temp/ logs/
   ```

2. **Enable debug logging**:
   ```bash
   # Increase log level
   export LOG_LEVEL=DEBUG
   
   # Enable request tracing
   export ENABLE_REQUEST_TRACING=true
   
   # Restart with debug settings
   docker-compose restart analog-pde-solver
   ```

---

## Escalation Procedures

### When to Escalate
- Critical issues not resolved within 30 minutes
- Data corruption suspected
- Security incident detected
- Multiple systems affected

### Escalation Contacts
1. **On-call Engineer**: Slack #oncall or phone
2. **System Architect**: For design/architectural issues
3. **Security Team**: For suspected security incidents
4. **Management**: For customer-impacting outages > 1 hour

### Information to Include
- Incident timeline
- Steps taken so far
- Current system status
- Customer impact assessment
- Recommended next steps

---

## Post-Incident Procedures

### Immediate (Within 2 hours)
- [ ] Restore full service
- [ ] Document timeline and actions
- [ ] Notify stakeholders of resolution
- [ ] Create post-incident review ticket

### Follow-up (Within 1 week)
- [ ] Conduct blameless post-mortem
- [ ] Identify root cause
- [ ] Create action items for prevention
- [ ] Update runbooks/procedures
- [ ] Share learnings with team

### Post-Mortem Template
```markdown
# Post-Incident Review: [Incident Title]

## Summary
- **Date**: YYYY-MM-DD
- **Duration**: X hours Y minutes
- **Severity**: Critical/High/Medium/Low
- **Impact**: Customer-facing/Internal

## Timeline
- HH:MM - Initial alert
- HH:MM - Response started
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Full resolution

## Root Cause
[Detailed analysis of what caused the incident]

## Actions Taken
[What was done to resolve the incident]

## What Went Well
[Positive aspects of the response]

## What Could Be Improved
[Areas for improvement]

## Action Items
- [ ] Item 1 (Owner: Name, Due: Date)
- [ ] Item 2 (Owner: Name, Due: Date)
```

---

## Emergency Contacts

- **On-call Engineer**: #oncall-alerts Slack channel
- **Platform Team**: platform-team@company.com
- **Security Team**: security@company.com
- **Management**: engineering-leads@company.com

## Useful Commands Reference

```bash
# Quick system health check
python monitoring/health-checks/health_check.py

# Service logs
docker logs -f analog-pde-solver --tail=100

# Resource usage
htop
iotop
nethogs

# Network diagnostics
ss -tulpn | grep :8000
netstat -i

# File system
df -h
lsof +D /path/to/directory

# Process management
ps aux | grep analog-pde
kill -9 PID
pkill -f analog-pde-solver
```