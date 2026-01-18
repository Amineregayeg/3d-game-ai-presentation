# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously at AASP. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email us at: **security@harbyx.com**
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Resolution Timeline**: We aim to resolve critical issues within 30 days
- **Credit**: We will credit you in our security advisories (unless you prefer anonymity)

### Scope

This security policy applies to:
- The AASP Platform (this repository)
- The AASP Python SDK (`aasp-sdk` package)
- The AASP API (`aasp-mvp.aminereg.com`)
- The AASP Dashboard

### Out of Scope

- Social engineering attacks
- Denial of service attacks
- Issues in third-party dependencies (report to maintainers directly)

## Security Features

AASP is built with security in mind:

- **API Key Hashing**: Keys are stored as SHA-256 hashes, not plaintext
- **HTTPS Enforcement**: All API communication is encrypted
- **Session Security**: httpOnly cookies with secure flags
- **Audit Logging**: All actions are logged for compliance
- **Input Validation**: All inputs are validated and sanitized
- **Rate Limiting**: Protection against abuse (coming soon)

## Security Best Practices for Users

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Store sensitive configuration in environment variables
3. **HTTPS**: Always use HTTPS endpoints
4. **Least Privilege**: Use test keys (`aasp_test_*`) during development
5. **Key Rotation**: Rotate API keys regularly

Thank you for helping keep AASP and its users safe!
