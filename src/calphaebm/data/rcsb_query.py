# src/calphaebm/data/rcsb_query.py

"""RCSB Search/Data API helpers to build nonredundant datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"


@dataclass(frozen=True)
class PolymerEntityInfo:
    """Information about a polymer entity from RCSB."""

    polymer_entity_id: str  # e.g., "1ABC_1"
    entry_id: str  # e.g., "1ABC"
    polymer_type: Optional[str]  # "Protein", "DNA", "RNA", etc.
    cluster_id_70: Optional[str]  # 70% sequence identity cluster ID
    resolution: Optional[float]  # Resolution in Å


def search_entries_xray_resolution(
    max_resolution: float = 2.0,
    page_size: int = 10000,
    start: int = 0,
    session: Optional[requests.Session] = None,
) -> Tuple[List[str], Optional[int]]:
    """Search for X-ray entries with resolution <= max_resolution.

    Args:
        max_resolution: Maximum resolution in Å.
        page_size: Number of results per page.
        start: Starting index.
        session: Optional requests session.

    Returns:
        (entry_ids, next_start) where next_start is None if no more results.
    """
    sess = session or requests.Session()

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": start, "rows": page_size},
            "sort": [
                {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}
            ],
        },
        "return_type": "entry",
    }

    resp = sess.post(SEARCH_URL, json=query, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    ids = [r["identifier"] for r in data.get("result_set", []) if "identifier" in r]
    total = data.get("total_count")

    next_start = start + page_size
    if total is None or next_start >= total:
        next_start = None

    return ids, next_start


def graphql_polymer_entities_for_entries(
    entry_ids: List[str],
    session: Optional[requests.Session] = None,
) -> List[PolymerEntityInfo]:
    """Fetch polymer entity information for entries.

    Args:
        entry_ids: List of PDB entry IDs.
        session: Optional requests session.

    Returns:
        List of PolymerEntityInfo objects.
    """
    if not entry_ids:
        return []

    sess = session or requests.Session()

    query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_entry_info {
          resolution_combined
        }
        polymer_entities {
          rcsb_id
          entity_poly {
            rcsb_entity_polymer_type
          }
          rcsb_cluster_membership {
            cluster_id
            identity
          }
        }
      }
    }
    """

    variables = {"ids": [eid.upper() for eid in entry_ids]}

    resp = sess.post(
        GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=60
    )
    resp.raise_for_status()
    payload = resp.json()

    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")

    entries = payload.get("data", {}).get("entries", []) or []

    out: List[PolymerEntityInfo] = []
    for ent in entries:
        entry_id = ent.get("rcsb_id")
        res = ent.get("rcsb_entry_info", {}).get("resolution_combined")

        for pe in ent.get("polymer_entities") or []:
            pe_id = pe.get("rcsb_id")
            poly_type = pe.get("entity_poly", {}).get("rcsb_entity_polymer_type")

            # Find 70% cluster
            cluster70 = None
            for cm in pe.get("rcsb_cluster_membership") or []:
                if cm.get("identity") == 70:
                    cluster70 = cm.get("cluster_id")
                    break

            out.append(
                PolymerEntityInfo(
                    polymer_entity_id=pe_id,
                    entry_id=entry_id,
                    polymer_type=poly_type,
                    cluster_id_70=cluster70,
                    resolution=res,
                )
            )

    return out


def is_protein_only_entry(pe_list: List[PolymerEntityInfo]) -> bool:
    """Check if all polymer entities in an entry are proteins."""
    if not pe_list:
        return False

    for pe in pe_list:
        if (pe.polymer_type or "").lower() != "protein":
            return False

    return True


def get_protein_entries(
    max_resolution: float = 2.0,
    max_entries: int = 10000,
    session: Optional[requests.Session] = None,
) -> List[str]:
    """Get list of protein-only entries meeting resolution criteria."""
    sess = session or requests.Session()

    all_entries = []
    start = 0
    page_size = 1000

    while len(all_entries) < max_entries:
        entries, next_start = search_entries_xray_resolution(
            max_resolution=max_resolution,
            page_size=page_size,
            start=start,
            session=sess,
        )

        if not entries:
            break

        # Check each entry for protein-only
        for i in range(0, len(entries), 100):  # Batch GraphQL calls
            batch = entries[i : i + 100]
            pe_list = graphql_polymer_entities_for_entries(batch, session=sess)

            # Group by entry
            by_entry = {}
            for pe in pe_list:
                by_entry.setdefault(pe.entry_id, []).append(pe)

            for eid, plist in by_entry.items():
                if is_protein_only_entry(plist) and eid not in all_entries:
                    all_entries.append(eid)
                    if len(all_entries) >= max_entries:
                        break

        if next_start is None:
            break
        start = next_start

    return all_entries[:max_entries]
