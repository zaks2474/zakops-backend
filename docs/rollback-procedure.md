# ZakOps Backend Rollback Procedure

**Phase 14: Deployment**

This document provides step-by-step instructions for rolling back the ZakOps backend to a previous version.

## When to Rollback

Initiate a rollback when:
- Critical bugs are discovered after deployment
- Performance degradation beyond acceptable thresholds
- Security vulnerabilities identified
- Integration failures with dependent services
- Health checks consistently failing

## Pre-Rollback Checklist

Before rolling back:
- [ ] Identify the target rollback version
- [ ] Confirm the issue is caused by the new deployment
- [ ] Notify stakeholders
- [ ] Document the issue for post-mortem

## Rollback Procedures

### Docker Compose Rollback

#### Quick Rollback (Same Machine, Previous Image Available)

```bash
cd /path/to/zakops-backend/infra/docker

# Step 1: Identify current and target versions
docker images zakops-backend

# Step 2: Stop current services
docker-compose -f docker-compose.production.yml down

# Step 3: Deploy previous version
export VERSION=<previous-version>
docker-compose -f docker-compose.production.yml up -d

# Step 4: Verify rollback
curl http://localhost:8000/api/version
curl http://localhost:8000/health
```

#### Full Rollback (Rebuild from Git Tag)

```bash
cd /path/to/zakops-backend

# Step 1: Identify target release
git tag --list

# Step 2: Checkout target version
git checkout v<version>

# Step 3: Rebuild and deploy
cd infra/docker
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d

# Step 4: Verify rollback
curl http://localhost:8000/api/version
curl http://localhost:8000/health
```

### Database Rollback

**Important**: Database rollbacks are more complex and may result in data loss.

#### Schema-Only Rollback

If the new version added schema changes that need to be reverted:

```sql
-- Example: Remove added column
ALTER TABLE zakops.deals DROP COLUMN IF EXISTS new_column;

-- Example: Revert enum changes
-- (Requires careful handling - may need to drop and recreate)
```

#### Full Database Restore

Only use for critical failures. Requires downtime.

```bash
# Step 1: Stop all services
docker-compose -f docker-compose.production.yml down

# Step 2: Restore from backup
# Using pg_restore (adjust for your backup solution)
pg_restore -h $DB_HOST -U $DB_USER -d zakops backup_file.dump

# Step 3: Start services with rolled-back version
export VERSION=<previous-version>
docker-compose -f docker-compose.production.yml up -d
```

## Rollback Scenarios

### Scenario 1: API Breaking Changes

**Symptoms**: 4xx/5xx errors from clients, integration failures

**Procedure**:
1. Quick rollback to previous API version
2. No database changes needed (usually)
3. Notify API consumers

### Scenario 2: Performance Regression

**Symptoms**: Slow response times, high CPU/memory

**Procedure**:
1. Quick rollback to previous version
2. Capture metrics/logs before rollback for analysis
3. Scale down if using multiple instances

### Scenario 3: Database Migration Failure

**Symptoms**: Migration script errors, data integrity issues

**Procedure**:
1. Stop all services immediately
2. Assess data impact
3. If data is intact: rollback application code only
4. If data is corrupted: restore from backup (requires decision on data loss)

### Scenario 4: Security Vulnerability

**Symptoms**: Security scan findings, exploit attempts

**Procedure**:
1. Immediate rollback to last known secure version
2. Block affected endpoints at load balancer if needed
3. Audit logs for potential breaches
4. Follow security incident procedure

## Post-Rollback Steps

After completing the rollback:

1. **Verify Health**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Version check
   curl http://localhost:8000/api/version

   # Check logs for errors
   docker logs zakops-backend --tail 100
   ```

2. **Monitor Metrics**
   - Watch error rates for 15-30 minutes
   - Verify SSE connections are restored
   - Check outbox processing is working

3. **Communicate**
   - Update status page if applicable
   - Notify affected teams
   - Send all-clear when stable

4. **Document**
   - Create incident report
   - Document root cause (when identified)
   - Update runbook if needed

## Version History

Maintain a record of deployments for quick reference:

| Version | Deploy Date | Git Commit | Notes |
|---------|-------------|------------|-------|
| 1.0.0   | 2026-01-19  | abc1234    | Initial release |
| ...     | ...         | ...        | ... |

## Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call | [PagerDuty] | Immediate |
| Tech Lead | [Contact] | 15 min |
| Database Admin | [Contact] | If DB rollback needed |

## Rollback Decision Tree

```
Issue Detected
     │
     ▼
Is service completely down?
     │
     ├─ Yes ──► Immediate rollback
     │
     └─ No ──► Is data being corrupted?
                    │
                    ├─ Yes ──► Stop services, assess, rollback
                    │
                    └─ No ──► Is it affecting >10% of users?
                                   │
                                   ├─ Yes ──► Rollback within 30 min
                                   │
                                   └─ No ──► Investigate, fix-forward if <1hr
```

## Prevention

To minimize rollback needs:
- Run comprehensive tests before deployment
- Use feature flags for gradual rollouts
- Deploy during low-traffic periods
- Have monitoring alerts configured
- Practice rollbacks in staging environment
