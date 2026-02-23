# src/calphaebm/cli/commands/build_dataset.py

"""Build PDB70-like dataset command."""

import argparse
import json

from calphaebm.data.build_pdb70_like import build_pdb70_like_polymer_entities
from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers):
    """Add build-dataset command parser."""
    parser = subparsers.add_parser(
        "build-dataset",
        description="Build PDB70-like nonredundant dataset",
        help="Build dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--target",
        type=int,
        default=10000,
        help="Target number of entities (default: 10000)",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Max resolution in A (default: 2.0)",
    )

    parser.add_argument(
        "--out-entities",
        default="pdb70_like_entities.txt",
        help="Output file for entity IDs (default: pdb70_like_entities.txt)",
    )

    parser.add_argument(
        "--out-entries",
        default="pdb70_like_entries.txt",
        help="Output file for entry IDs (default: pdb70_like_entries.txt)",
    )

    parser.add_argument(
        "--meta",
        default="pdb70_like_meta.json",
        help="Output file for metadata (default: pdb70_like_meta.json)",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=5000,
        help="Search API page size (default: 5000)",
    )

    parser.add_argument(
        "--graphql-batch",
        type=int,
        default=200,
        help="GraphQL batch size (default: 200)",
    )

    parser.set_defaults(func=run)


def run(args):
    """Run build-dataset command."""
    logger.info(f"Building PDB70-like dataset with target {args.target}")
    logger.info(f"Max resolution: {args.resolution} A")

    result = build_pdb70_like_polymer_entities(
        target_n=args.target,
        max_resolution=args.resolution,
        page_size=args.page_size,
        graphql_batch=args.graphql_batch,
        verbose=True,
    )

    # Save entity IDs
    with open(args.out_entities, "w") as f:
        for pid in result.polymer_entity_ids:
            f.write(pid + "\n")

    # Save entry IDs
    entry_ids = sorted(
        {pid.split("_")[0].upper() for pid in result.polymer_entity_ids if pid}
    )

    with open(args.out_entries, "w") as f:
        for eid in entry_ids:
            f.write(eid + "\n")

    # Save metadata
    meta = {
        "target": args.target,
        "resolution": args.resolution,
        "n_selected_entities": len(result.polymer_entity_ids),
        "n_unique_entries": len(entry_ids),
        "n_candidates_entries_seen": result.n_candidates_entries_seen,
        "n_unique_clusters": result.n_unique_clusters,
    }

    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Saved {len(result.polymer_entity_ids)} entities to {args.out_entities}"
    )
    logger.info(f"Saved {len(entry_ids)} entries to {args.out_entries}")
    logger.info(f"Metadata: {args.meta}")

    return 0
