# src/calphaebm/cli/commands/balance.py

"""Lambda balancing command."""

import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from calphaebm.data.pdb_dataset import PDBSegmentDataset
from calphaebm.models.cross_terms import SecondaryStructureEnergy
from calphaebm.models.energy import TotalEnergy
from calphaebm.models.local_terms import LocalEnergy
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import RepulsionEnergy
from calphaebm.training.balancing import recommend_lambdas
from calphaebm.utils.constants import EMB_DIM, NUM_AA
from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers):
    """Add balance command parser."""
    parser = subparsers.add_parser(
        "balance",
        description="Recommend lambda weights by balancing force scales",
        help="Balance force scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdb",
        nargs="+",
        required=True,
        help="PDB IDs or file containing IDs",
    )

    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path",
    )

    parser.add_argument(
        "--cache-dir",
        default="./pdb_cache",
        help="PDB cache directory",
    )

    parser.add_argument(
        "--seg-len",
        type=int,
        default=64,
        help="Segment length (default: 64)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Segment stride (default: 32)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=512,
        help="Maximum segments to load (default: 512)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )

    parser.add_argument(
        "--energy-terms",
        nargs="*",
        default=["local", "repulsion", "secondary", "packing"],
        choices=["local", "repulsion", "secondary", "packing", "all"],
        help="Energy terms to include",
    )

    parser.add_argument(
        "--reference",
        default="local",
        choices=["local", "repulsion", "secondary", "packing"],
        help="Reference term for scaling (default: local)",
    )

    parser.add_argument(
        "--out",
        help="Output JSON file for results",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Print suggested lambda flags",
    )

    parser.set_defaults(func=run)


def _read_id_lines(path: str) -> List[str]:
    """Read IDs from file."""
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def _normalize_ids(raw: List[str]) -> List[str]:
    """Normalize PDB IDs."""
    ids = [x.split("_")[0].upper() for x in raw]
    seen = set()
    out = []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_pdb_arg(values: List[str]) -> List[str]:
    """Parse PDB argument (list of IDs or file path)."""
    if len(values) == 1 and os.path.exists(values[0]) and os.path.isfile(values[0]):
        return _normalize_ids(_read_id_lines(values[0]))
    return _normalize_ids(values)


def run(args):
    """Run balance command."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse PDB IDs
    pdb_ids = parse_pdb_arg(args.pdb)
    if not pdb_ids:
        logger.error("No valid PDB IDs found")
        return 1

    # Parse terms
    terms_set = {t.lower() for t in args.energy_terms}
    if "all" in terms_set:
        terms_set = {"local", "repulsion", "secondary", "packing"}
    terms_set.add("local")

    # Load dataset
    ds = PDBSegmentDataset(
        pdb_ids=pdb_ids,
        cache_dir=args.cache_dir,
        seg_len=args.seg_len,
        stride=args.stride,
        limit_segments=args.limit,
    )

    if len(ds) == 0:
        logger.error("No segments found")
        return 1

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Get a batch
    coords, seq, _, _ = next(iter(dl))
    coords = coords.to(device)
    seq = seq.to(device)

    # Build model
    local = LocalEnergy(
        num_aa=NUM_AA,
        emb_dim=EMB_DIM,
        hidden=(128, 128),
        use_cos_theta=True,
    ).to(device)

    rep = RepulsionEnergy().to(device) if "repulsion" in terms_set else None
    ss = (
        SecondaryStructureEnergy(
            num_aa=NUM_AA,
            emb_dim=EMB_DIM,
        ).to(device)
        if "secondary" in terms_set
        else None
    )
    pack = (
        PackingEnergy(
            num_aa=NUM_AA,
            emb_dim=EMB_DIM,
        ).to(device)
        if "packing" in terms_set
        else None
    )

    model = TotalEnergy(
        local=local,
        repulsion=rep,
        secondary=ss,
        packing=pack,
    ).to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    if "gates" in ckpt:
        model.set_gates(**ckpt["gates"])

    model.eval()

    # Compute forces and recommend lambdas
    logger.info("Computing force scales...")

    report = recommend_lambdas(
        model,
        coords,
        seq,
        reference=args.reference,
        current_lambdas=None,
    )

    # Print results
    print("\nForce scales (median |F| per term):")
    for term, scale in report.force_scales.items():
        print(f"  {term:12s}: {scale:.6f}")

    print("\nRecommended lambdas:")
    for term, lam in report.recommended_lambdas.items():
        print(f"  lambda-{term:12s}: {lam:.6f}")

    if args.apply:
        rec = report.recommended_lambdas
        print("\nCommand-line flags:")
        print(
            f"  --lambda-local {rec.get('local', 1.0):.6f} "
            f"--lambda-rep {rec.get('repulsion', 1.0):.6f} "
            f"--lambda-ss {rec.get('secondary', 1.0):.6f} "
            f"--lambda-pack {rec.get('packing', 1.0):.6f}"
        )

    # Save to file
    if args.out:
        output = {
            "reference": args.reference,
            "force_scales": report.force_scales,
            "recommended_lambdas": report.recommended_lambdas,
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {args.out}")

    return 0
