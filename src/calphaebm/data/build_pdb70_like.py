# src/calphaebm/data/build_pdb70_like.py

"""Build a PDB70-like nonredundant set of polymer entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Dict
import requests
from tqdm import tqdm

from calphaebm.data.rcsb_query import (
    search_entries_xray_resolution,
    graphql_polymer_entities_for_entries,
    is_protein_only_entry,
    PolymerEntityInfo,
)
from calphaebm.utils.logging import get_logger

logger = get_logger()


@dataclass
class BuildResult:
    """Result of building a PDB70-like set."""
    
    polymer_entity_ids: List[str]
    n_candidates_entries_seen: int
    n_unique_clusters: int


def build_pdb70_like_polymer_entities(
    target_n: int = 10000,
    max_resolution: float = 2.0,
    page_size: int = 5000,
    graphql_batch: int = 200,
    session: Optional[requests.Session] = None,
    verbose: bool = True,
) -> BuildResult:
    """Build a nonredundant set of protein polymer entities.
    
    Selects one polymer entity per 70% sequence identity cluster,
    prioritizing higher resolution.
    
    Args:
        target_n: Target number of polymer entities.
        max_resolution: Maximum resolution in Å.
        page_size: Page size for search API.
        graphql_batch: Batch size for GraphQL queries.
        session: Optional requests session.
        verbose: Show progress bar.
        
    Returns:
        BuildResult with selected polymer entity IDs.
    """
    sess = session or requests.Session()
    
    seen_clusters: Set[str] = set()
    selected: List[str] = []
    entries_seen = 0
    start = 0
    
    pbar = tqdm(total=target_n, disable=not verbose, desc="Selecting entities")
    
    while len(selected) < target_n:
        entry_ids, next_start = search_entries_xray_resolution(
            max_resolution=max_resolution,
            page_size=page_size,
            start=start,
            session=sess,
        )
        
        if not entry_ids:
            break
            
        entries_seen += len(entry_ids)
        
        # Process in batches
        for b0 in range(0, len(entry_ids), graphql_batch):
            if len(selected) >= target_n:
                break
                
            batch_ids = entry_ids[b0:b0 + graphql_batch]
            pe_infos = graphql_polymer_entities_for_entries(batch_ids, session=sess)
            
            # Group by entry
            by_entry: Dict[str, List[PolymerEntityInfo]] = {}
            for pe in pe_infos:
                by_entry.setdefault(pe.entry_id, []).append(pe)
            
            # Process each entry
            for entry_id, pe_list in by_entry.items():
                if len(selected) >= target_n:
                    break
                    
                if not is_protein_only_entry(pe_list):
                    continue
                
                # Sort polymer entities by resolution (best first)
                pe_list.sort(key=lambda x: x.resolution if x.resolution else float("inf"))
                
                for pe in pe_list:
                    if len(selected) >= target_n:
                        break
                        
                    if pe.cluster_id_70 is None:
                        continue
                        
                    if pe.cluster_id_70 in seen_clusters:
                        continue
                    
                    seen_clusters.add(pe.cluster_id_70)
                    selected.append(pe.polymer_entity_id)
                    pbar.update(1)
        
        if next_start is None:
            break
        start = next_start
    
    pbar.close()
    
    logger.info(f"Selected {len(selected)} entities from {len(seen_clusters)} clusters")
    logger.info(f"Scanned {entries_seen} entries")
    
    return BuildResult(
        polymer_entity_ids=selected,
        n_candidates_entries_seen=entries_seen,
        n_unique_clusters=len(seen_clusters),
    )