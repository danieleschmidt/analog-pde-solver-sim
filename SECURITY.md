# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please report it to us as follows:

### Private Disclosure

1. **Email**: Send details to security@yourdomain.com
2. **Subject**: "Security Vulnerability in analog-pde-solver-sim"
3. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity (1-4 weeks)

### Security Considerations

This project involves:
- Hardware simulation (SPICE, Verilog)
- Mathematical computing with potential numerical instabilities
- File I/O operations for simulation data

Common security considerations:
- Input validation for PDE parameters
- Safe handling of simulation output files
- Memory management in large matrix operations
- Sanitization of user-provided equations

### Safe Usage Guidelines

- Always validate input parameters before simulation
- Use virtual environments for dependency isolation
- Be cautious with SPICE netlists from untrusted sources
- Monitor system resources during large simulations

Thank you for helping keep analog-pde-solver-sim secure!