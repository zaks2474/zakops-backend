"""
Search API

Endpoints for searching deals and actions with cursor-based pagination.
"""

from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import base64
import json

from ....core.database.adapter import get_database

router = APIRouter(prefix="/api/search", tags=["search"])


class PaginationCursor(BaseModel):
    """Opaque cursor for stable pagination."""
    last_id: str
    last_timestamp: str

    def encode(self) -> str:
        """Encode cursor to URL-safe string."""
        data = {"id": self.last_id, "ts": self.last_timestamp}
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

    @classmethod
    def decode(cls, cursor: str) -> "PaginationCursor":
        """Decode cursor from URL-safe string."""
        try:
            data = json.loads(base64.urlsafe_b64decode(cursor))
            return cls(last_id=data["id"], last_timestamp=data["ts"])
        except Exception:
            raise ValueError("Invalid cursor format")


class DealSearchResult(BaseModel):
    """Deal search result."""
    deal_id: str
    canonical_name: str
    company_name: Optional[str] = None
    stage: str
    status: str
    created_at: str
    updated_at: Optional[str] = None


class DealSearchResponse(BaseModel):
    """Response from deal search."""
    results: List[DealSearchResult]
    next_cursor: Optional[str] = None
    has_more: bool = False
    query: str
    # Legacy fields for backwards compatibility
    total: Optional[int] = None
    limit: int
    offset: Optional[int] = None


class ActionSearchResult(BaseModel):
    """Action search result."""
    action_id: str
    deal_id: Optional[str] = None
    action_type: str
    title: Optional[str] = None
    status: str
    risk_level: str
    created_at: str


class ActionSearchResponse(BaseModel):
    """Response from action search."""
    results: List[ActionSearchResult]
    next_cursor: Optional[str] = None
    has_more: bool = False
    query: str
    limit: int


class GlobalSearchResult(BaseModel):
    """Global search result item."""
    id: str
    type: str  # "deal" or "action"
    title: str
    subtitle: Optional[str] = None


@router.get("/deals", response_model=DealSearchResponse)
async def search_deals(
    q: str = Query(..., min_length=1, description="Search query"),
    stage: Optional[str] = Query(None, description="Filter by stage"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor from previous response"),
    offset: Optional[int] = Query(None, ge=0, description="Legacy offset (use cursor instead)")
):
    """
    Search deals by name, company, or content with cursor-based pagination.

    Cursor pagination ensures stable results even when data changes between requests.
    Use the next_cursor from the response in subsequent requests to get the next page.

    Example:
        # First request
        GET /api/search/deals?q=acme&limit=10

        # Next page (use next_cursor from response)
        GET /api/search/deals?q=acme&limit=10&cursor=eyJpZCI6...
    """
    db = await get_database()

    conditions = ["deleted = FALSE"]
    params = []
    param_count = 0

    # Text search
    param_count += 1
    conditions.append(f"""
        (canonical_name ILIKE ${param_count}
         OR display_name ILIKE ${param_count}
         OR CAST(company_info AS TEXT) ILIKE ${param_count})
    """)
    params.append(f"%{q}%")

    # Stage filter
    if stage:
        param_count += 1
        conditions.append(f"stage = ${param_count}")
        params.append(stage)

    # Status filter
    if status:
        param_count += 1
        conditions.append(f"status = ${param_count}")
        params.append(status)

    # Cursor-based pagination (keyset pagination)
    if cursor:
        try:
            c = PaginationCursor.decode(cursor)
            param_count += 1
            ts_param = param_count
            param_count += 1
            id_param = param_count
            conditions.append(f"""
                (updated_at, deal_id) < (${ts_param}::timestamptz, ${id_param})
            """)
            params.extend([c.last_timestamp, c.last_id])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid cursor format")

    where_clause = " AND ".join(conditions)

    # Query with stable ordering (updated_at DESC, deal_id DESC)
    # Fetch one extra to determine has_more
    param_count += 1
    query = f"""
        SELECT deal_id, canonical_name, display_name, stage, status, created_at, updated_at,
               company_info->>'name' as company_name
        FROM zakops.deals
        WHERE {where_clause}
        ORDER BY updated_at DESC, deal_id DESC
        LIMIT ${param_count}
    """
    params.append(limit + 1)

    results = await db.fetch(query, *params)

    # Determine if there are more results
    has_more = len(results) > limit
    if has_more:
        results = results[:limit]

    # Generate next cursor from last result
    next_cursor = None
    if has_more and results:
        last = results[-1]
        next_cursor = PaginationCursor(
            last_id=str(last["deal_id"]),
            last_timestamp=last["updated_at"].isoformat() if last.get("updated_at") else ""
        ).encode()

    return DealSearchResponse(
        results=[
            DealSearchResult(
                deal_id=r["deal_id"],
                canonical_name=r["canonical_name"],
                company_name=r.get("company_name") or r.get("display_name"),
                stage=r["stage"],
                status=r["status"],
                created_at=r["created_at"].isoformat() if r.get("created_at") else "",
                updated_at=r["updated_at"].isoformat() if r.get("updated_at") else None
            )
            for r in results
        ],
        next_cursor=next_cursor,
        has_more=has_more,
        query=q,
        limit=limit,
        offset=offset
    )


@router.get("/actions", response_model=ActionSearchResponse)
async def search_actions(
    q: str = Query(..., min_length=1, description="Search query"),
    status: Optional[str] = Query(None, description="Filter by status"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    deal_id: Optional[str] = Query(None, description="Filter by deal ID"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor")
):
    """
    Search actions with cursor-based pagination.
    """
    db = await get_database()

    conditions = []
    params = []
    param_count = 0

    # Text search
    param_count += 1
    conditions.append(f"""
        (action_type ILIKE ${param_count}
         OR title ILIKE ${param_count}
         OR CAST(inputs AS TEXT) ILIKE ${param_count})
    """)
    params.append(f"%{q}%")

    if status:
        param_count += 1
        conditions.append(f"status = ${param_count}")
        params.append(status)

    if action_type:
        param_count += 1
        conditions.append(f"action_type = ${param_count}")
        params.append(action_type)

    if deal_id:
        param_count += 1
        conditions.append(f"deal_id = ${param_count}")
        params.append(deal_id)

    # Cursor pagination
    if cursor:
        try:
            c = PaginationCursor.decode(cursor)
            param_count += 1
            ts_param = param_count
            param_count += 1
            id_param = param_count
            conditions.append(f"""
                (created_at, action_id) < (${ts_param}::timestamptz, ${id_param})
            """)
            params.extend([c.last_timestamp, c.last_id])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid cursor format")

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    param_count += 1
    query = f"""
        SELECT action_id, deal_id, action_type, title, status, risk_level, created_at
        FROM zakops.actions
        WHERE {where_clause}
        ORDER BY created_at DESC, action_id DESC
        LIMIT ${param_count}
    """
    params.append(limit + 1)

    results = await db.fetch(query, *params)

    has_more = len(results) > limit
    if has_more:
        results = results[:limit]

    next_cursor = None
    if has_more and results:
        last = results[-1]
        next_cursor = PaginationCursor(
            last_id=str(last["action_id"]),
            last_timestamp=last["created_at"].isoformat() if last.get("created_at") else ""
        ).encode()

    return ActionSearchResponse(
        results=[
            ActionSearchResult(
                action_id=r["action_id"],
                deal_id=r.get("deal_id"),
                action_type=r["action_type"],
                title=r.get("title"),
                status=r["status"],
                risk_level=r.get("risk_level", "low"),
                created_at=r["created_at"].isoformat() if r.get("created_at") else ""
            )
            for r in results
        ],
        next_cursor=next_cursor,
        has_more=has_more,
        query=q,
        limit=limit
    )


@router.get("/global")
async def global_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Results per type")
):
    """
    Search across all entity types.

    Returns top results from deals and actions without pagination
    (use specific endpoints for paginated results).
    """
    db = await get_database()
    search_pattern = f"%{q}%"

    # Search deals
    deals = await db.fetch(
        """
        SELECT deal_id as id, 'deal' as type, canonical_name as title,
               COALESCE(display_name, stage) as subtitle
        FROM zakops.deals
        WHERE deleted = FALSE
          AND (canonical_name ILIKE $1 OR display_name ILIKE $1)
        ORDER BY updated_at DESC
        LIMIT $2
        """,
        search_pattern, limit
    )

    # Search actions
    actions = await db.fetch(
        """
        SELECT action_id as id, 'action' as type,
               COALESCE(title, action_type) as title,
               status as subtitle
        FROM zakops.actions
        WHERE action_type ILIKE $1
           OR title ILIKE $1
           OR CAST(inputs AS TEXT) ILIKE $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        search_pattern, limit
    )

    return {
        "deals": [
            {
                "id": d["id"],
                "type": "deal",
                "title": d["title"],
                "subtitle": d.get("subtitle")
            }
            for d in deals
        ],
        "actions": [
            {
                "id": a["id"],
                "type": "action",
                "title": a["title"],
                "subtitle": a.get("subtitle")
            }
            for a in actions
        ],
        "query": q
    }
