# Autonomous SDLC Setup Guide

## Overview

This repository has been enhanced with **Terragon Autonomous SDLC** capabilities that continuously discover, prioritize, and execute high-value development tasks. This guide covers the setup and operation of the autonomous system.

## Prerequisites

```bash
# Install Claude Code and Claude-Flow
npm i -g @anthropic-ai/claude-code
npm i -g claude-flow@alpha

# Configure Claude Code
claude --dangerously-skip-permissions
```

## Autonomous Execution

### Initial Setup

```bash
# Clone and setup
git clone <repository-url>
cd analog-pde-solver-sim

# Install dependencies
make dev-setup

# Verify autonomous configuration
ls .terragon/
```

### Continuous Value Discovery

The autonomous system operates on multiple schedules:

- **Every PR merge**: Immediate value discovery and next item selection
- **Hourly**: Security and dependency vulnerability scans  
- **Daily**: Comprehensive static analysis and technical debt assessment
- **Weekly**: Deep architectural analysis and modernization opportunities
- **Monthly**: Strategic value alignment and scoring model recalibration

### Manual Trigger

Execute autonomous enhancement cycle:

```bash
npx claude-flow@alpha swarm "AUTONOMOUS SDLC enhancement for repository ${PWD##*/}:

Analyze current repository state, discover high-value work items using configured scoring (WSJF + ICE + Technical Debt), and execute the highest-priority item.

Focus on:
- Technical debt reduction in high-churn files
- Security vulnerability remediation  
- Performance optimization opportunities
- Documentation gaps in critical paths
- Infrastructure modernization

Update .terragon/value-metrics.json with execution results and discover next best value item.
" --strategy autonomous --claude
```

## Configuration Files

### `.terragon/config.yaml`
- Repository maturity classification (MATURING)
- Scoring weights optimized for current development phase
- Discovery sources and tools configuration
- Execution parameters and rollback triggers

### `.terragon/value-metrics.json`
- Current repository assessment and capabilities
- Discovered value items with scoring
- Execution history and learning metrics
- Backlog statistics and trends

### `.terragon/discovery-rules.yaml`
- Pattern matching rules for different issue types
- Scoring algorithms for prioritization
- Discovery schedules and automation triggers

## Value Scoring System

### WSJF (Weighted Shortest Job First)
```
Cost of Delay = User Business Value + Time Criticality + Risk Reduction + Opportunity Enablement
WSJF Score = Cost of Delay / Job Size
```

### ICE (Impact, Confidence, Ease)
```
ICE Score = Impact × Confidence × Ease
```

### Technical Debt Scoring
```
Debt Score = (Debt Impact + Debt Interest) × Hotspot Multiplier
```

### Composite Score
```
Final Score = (0.6 × WSJF) + (0.1 × ICE) + (0.2 × Debt Score) + (0.1 × Security Boost)
```

## Monitoring and Metrics

### Real-time Status
```bash
# Check current backlog
cat BACKLOG.md

# View metrics dashboard
cat .terragon/value-metrics.json | jq '.backlog_metrics'

# Review execution history
cat .terragon/value-metrics.json | jq '.execution_history'
```

### Learning and Adaptation

The system continuously learns from execution outcomes:

- **Estimation Accuracy**: Tracks predicted vs actual effort
- **Value Delivery**: Measures actual business/technical impact
- **Pattern Recognition**: Improves issue detection over time
- **Scoring Calibration**: Adjusts weights based on results

## Integration with Development Workflow

### Pre-commit Integration
```bash
# Install pre-commit hooks (includes autonomous value discovery)
pre-commit install

# Run manual discovery
pre-commit run autonomous-discovery --all-files
```

### GitHub Integration
The system can integrate with GitHub APIs for:
- Issue and PR analysis
- Automated backlog updates
- Code review insights
- Release planning

## Troubleshooting

### Common Issues

1. **No items discovered**: Check discovery patterns in `.terragon/discovery-rules.yaml`
2. **Low scoring items**: Adjust weights in `.terragon/config.yaml`
3. **Execution failures**: Review rollback triggers and test requirements

### Logs and Debugging
```bash
# Check discovery logs
grep "autonomous" .git/hooks/pre-commit.log

# Validate configuration
python -c "import yaml; yaml.safe_load(open('.terragon/config.yaml'))"

# Test scoring
python scripts/test-scoring.py
```

## Customization

### Adjusting for Repository Maturity
As the repository evolves, update maturity classification:

```yaml
# .terragon/config.yaml
repository:
  maturity_level: "advanced"  # nascent -> developing -> maturing -> advanced

scoring:
  weights:
    # Advanced repositories: higher focus on optimization
    wsjf: 0.5
    technicalDebt: 0.3
    security: 0.1
    ice: 0.1
```

### Custom Discovery Patterns
Add domain-specific patterns:

```yaml
# .terragon/discovery-rules.yaml
patterns:
  analog_specific:
    spice_issues:
      - pattern: "convergence.error|simulation.failed"
        weight: 0.9
        category: "hardware_validation"
        
    verilog_issues:
      - pattern: "synthesis.error|timing.violation"
        weight: 0.8
        category: "rtl_generation"
```

## Best Practices

1. **Regular Review**: Weekly review of autonomous decisions and outcomes
2. **Manual Override**: Always maintain ability to manually prioritize critical items
3. **Feedback Loop**: Update scoring based on actual business impact
4. **Collaboration**: Share autonomous insights with team members
5. **Continuous Learning**: Monitor and improve discovery patterns

The autonomous system enhances development velocity while maintaining code quality and security standards.