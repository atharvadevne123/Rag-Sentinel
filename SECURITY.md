# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | yes       |
| 1.0.x   | yes       |
| < 1.0   | no        |

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Email devneatharva@gmail.com with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact and affected versions

You will receive a response within 48 hours. Confirmed vulnerabilities will be patched and a new release issued within 7 days.

## Known Attack Surface

RAG Sentinel performs query anomaly detection. Key areas to review:

| Component | Risk | Mitigation |
|-----------|------|------------|
| `/predict` input | Prompt injection, oversized input | Max 2000 chars, control-char sanitisation |
| `/ingest` text | Memory exhaustion | Max body size via `MAX_INGEST_BYTES` |
| `/ingest` doc_id | Path traversal | Alphanumeric + dash/dot/underscore only |
| DB connection | Credential exposure | Credentials via env vars only |
| Model files | Supply chain | Joblib files should be verified before loading |

## Dependency Scanning

Run `make audit` to check for known CVEs in dependencies via `pip-audit`.
