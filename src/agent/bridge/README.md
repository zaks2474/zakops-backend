# ZakOps Agent Bridge

MCP Server connecting LangSmith Agent Builder to local ZakOps infrastructure.

## Architecture

```
LangSmith Agent Builder (Cloud)
           │
           ▼
   Cloudflare Tunnel
           │
           ▼
ZakOps Agent Bridge (:9100)
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  Deal   RAG   DataRoom
  API   API   Filesystem
(:8090)(:8052)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Generate API key
   echo "ZAKOPS_BRIDGE_API_KEY=$(openssl rand -hex 32)" >> .env
   ```

3. **Start the server:**
   ```bash
   uvicorn mcp_server:app --host 127.0.0.1 --port 9100
   ```

## API Reference

### Health Check
```bash
curl http://localhost:9100/health
```

### List Available Tools
```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:9100/tools
```

### List Deals
```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:9100/tools/zakops/list_deals
```

### Get Deal
```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:9100/tools/zakops/get_deal/DEAL-2026-001
```

### Create Action
```bash
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"DEAL.UPDATE_STAGE","title":"Test","inputs":{}}' \
  http://localhost:9100/tools/zakops/create_action
```

### Query RAG
```bash
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"revenue","top_k":5}' \
  http://localhost:9100/tools/rag/query_local
```

## Security

- All endpoints except `/health` require Bearer token authentication
- Path traversal attacks are blocked on all file operations
- Atomic writes with verification for data integrity
- No delete operations permitted
- Email sending is draft-only

## Logs

Structured JSON logs are written to:
```
/home/zaks/DataRoom/.deal-registry/logs/agent_bridge.jsonl
```

## Systemd Service

Install and enable the service:
```bash
sudo cp /path/to/zakops-agent-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable zakops-agent-bridge
sudo systemctl start zakops-agent-bridge
```
