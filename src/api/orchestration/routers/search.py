"""
Search API

Endpoints for searching deals and actions.
"""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

from ....core.database.adapter import get_database

router = APIRouter(prefix="/api/search", tags=["search"])


class DealSearchResult(BaseModel):
    """Deal search result."""
    deal_id: str
    canonical_name: str
    company_name: Optional[str] = None
    stage: str
    status: str
    created_at: str


class ActionSearchResult(BaseModel):
    """Action search result."""
    action_id: str
    deal_id: Optional[str] = None
    action_type: str
    status: str
    risk_level: str
    created_at: str


class GlobalSearchResult(BaseModel):
    """Global search result item."""
    id: str
    type: str  # "deal" or "action"
    title: str
    subtitle: Optional[str] = None


@router.get("/deals")
async def search_deals(
    q: str = Query(..., min_length=1, description="Search query"),
    stage: Optional[str] = Query(None, description="Filter by stage"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Search deals by name, company, or content.
    """
    db = await get_database()

    # Build query
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

    where_clause = " AND ".join(conditions)

    # Execute query
    query = f"""
        SELECT deal_id, canonical_name, display_name, stage, status, created_at,
               company_info->>'name' as company_name
        FROM zakops.deals
        WHERE {where_clause}
        ORDER BY updated_at DESC
        LIMIT ${param_count + 1} OFFSET ${param_count + 2}
    """
    params.extend([limit, offset])

    results = await db.fetch(query, *params)

    # Get total count
    count_query = f"SELECT COUNT(*) as cnt FROM zakops.deals WHERE {where_clause}"
    total_row = await db.fetchrow(count_query, *params[:-2])
    total = total_row["cnt"] if total_row else 0

    return {
        "results": [
            {
                "deal_id": r["deal_id"],
                "canonical_name": r["canonical_name"],
                "company_name": r.get("company_name") or r.get("display_name"),
                "stage": r["stage"],
                "status": r["status"],
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None
            }
            for r in results
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
        "query": q
    }


@router.get("/actions")
async def search_actions(
    q: str = Query(..., min_length=1),
    status: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    deal_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Search actions by type or content.
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

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    query = f"""
        SELECT action_id, deal_id, action_type, title, status, risk_level, created_at
        FROM zakops.actions
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count + 1} OFFSET ${param_count + 2}
    """
    params.extend([limit, offset])

    results = await db.fetch(query, *params)

    return {
        "results": [
            {
                "action_id": r["action_id"],
                "deal_id": r.get("deal_id"),
                "action_type": r["action_type"],
                "title": r.get("title"),
                "status": r["status"],
                "risk_level": r.get("risk_level", "low"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None
            }
            for r in results
        ],
        "limit": limit,
        "offset": offset,
        "query": q
    }


@router.get("/global")
async def global_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search across all entity types.
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
