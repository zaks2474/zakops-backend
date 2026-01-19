# ZakOps Security Checklist

## Pre-Production Checklist

### Authentication & Authorization
- [ ] AUTH_REQUIRED=true in production
- [ ] Session secret is strong (32+ random bytes)
- [ ] Session cookies are Secure + HttpOnly + SameSite=Strict
- [ ] CSRF protection enabled
- [ ] Rate limiting on auth endpoints (10/min)
- [ ] Password hashing uses bcrypt or argon2

### Network Security
- [ ] HTTPS only (HTTP redirects to HTTPS)
- [ ] CORS restricted to frontend domain only
- [ ] No sensitive data in URL parameters
- [ ] X-Frame-Options: DENY
- [ ] X-Content-Type-Options: nosniff
- [ ] X-XSS-Protection: 1; mode=block
- [ ] Referrer-Policy: strict-origin-when-cross-origin
- [ ] Content-Security-Policy configured

### Data Protection
- [ ] Database credentials in secrets manager (not env file)
- [ ] API keys not logged
- [ ] Error messages sanitized
- [ ] No PII in logs
- [ ] Input validation on all endpoints
- [ ] SQL injection protection (parameterized queries)
- [ ] XSS protection (output encoding)

### Infrastructure
- [ ] Database not publicly accessible
- [ ] Redis not publicly accessible
- [ ] S3 bucket not public
- [ ] Secrets rotated regularly
- [ ] Backup encryption enabled
- [ ] VPC configured correctly
- [ ] Security groups restrict access

### Monitoring & Alerting
- [ ] Authentication failures logged
- [ ] Rate limit violations logged
- [ ] DLQ growth alerts configured
- [ ] Error rate alerts configured
- [ ] Unusual traffic patterns alerting
- [ ] Security audit logging enabled

---

## Security Headers

All responses should include these headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

---

## Rate Limits

| Endpoint | Limit | Notes |
|----------|-------|-------|
| /api/auth/* | 10/min | Login, register, password reset |
| /api/* (general) | 60/min | All other endpoints |
| /api/events/stream | 5 connections/user | SSE connections |

---

## Session Security

| Setting | Value | Reason |
|---------|-------|--------|
| Secure | true | Only send over HTTPS |
| HttpOnly | true | Not accessible via JavaScript |
| SameSite | Strict | CSRF protection |
| Max-Age | 24 hours | Session expiry |

---

## Error Message Sanitization

Error messages MUST NOT include:
- File paths (e.g., `/home/user/app/src/module.py`)
- Line numbers
- Database connection strings
- API keys or secrets
- Internal IDs or UUIDs
- Stack traces in production

Example sanitization:
```python
# Bad: "Error in /home/app/src/db.py line 42: connection refused to postgresql://user:pass@host/db"
# Good: "Database connection error. Please try again later."
```

---

## Incident Response

### Authentication Breach
1. Revoke all sessions immediately
2. Rotate session secret
3. Force password resets
4. Audit access logs
5. Notify affected users

### DLQ Overflow
1. Pause outbox processor
2. Investigate root cause
3. Fix underlying issue
4. Manual retry or purge
5. Resume processor

### SSE Overload
1. Shed oldest connections
2. Enable degraded mode (polling)
3. Scale horizontally if needed
4. Investigate traffic source

### Data Breach
1. Contain the breach
2. Assess impact
3. Notify affected parties
4. Preserve evidence
5. Post-incident review

---

## Secure Development Practices

### Code Review Checklist
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] Output encoding for XSS prevention
- [ ] Parameterized queries for SQL
- [ ] Proper error handling
- [ ] Authentication/authorization checks
- [ ] Logging without sensitive data

### Dependency Management
- [ ] Dependencies are pinned
- [ ] Regular vulnerability scanning
- [ ] No known vulnerable packages
- [ ] Minimal dependency footprint

---

## Compliance Considerations

### Data Retention
- Define retention periods for each data type
- Implement automated purging
- Document deletion procedures

### Audit Trail
- Log all admin actions
- Log all data modifications
- Retain logs for compliance period

### Access Control
- Principle of least privilege
- Regular access reviews
- Prompt deprovisioning

---

## Production Checklist Summary

Before deploying to production:

1. **Environment Variables**
   - [ ] All secrets are set
   - [ ] AUTH_REQUIRED=true
   - [ ] DEBUG=false
   - [ ] CORS origins restricted

2. **Database**
   - [ ] Migrations applied
   - [ ] Backups configured
   - [ ] Monitoring enabled

3. **Networking**
   - [ ] HTTPS certificates valid
   - [ ] Load balancer configured
   - [ ] Health checks working

4. **Monitoring**
   - [ ] Logs streaming to aggregator
   - [ ] Metrics being collected
   - [ ] Alerts configured

5. **Testing**
   - [ ] All tests passing
   - [ ] Load testing completed
   - [ ] Security scan passed
