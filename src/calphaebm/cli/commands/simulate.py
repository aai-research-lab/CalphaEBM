# src/calphaebm/cli/commands/simulate.py

"""Simulation command."""

import argparse
from pathlib import Path

import torch

from calphaebm.data.pdb_parse import (
    download_cif,
    get_residue_sequence,
    parse_cif_ca_chains,
)
from calphaebm.models.cross_terms import SecondaryStructureEnergy
from calphaebm.models.energy import TotalEnergy
from calphaebm.models.local_terms import LocalEnergy
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.repulsion import (
    RepulsionEnergyFixed,
    RepulsionEnergyLearnedRadius,
)
from calphaebm.simulation.backends.pytorch import PyTorchSimulator
from calphaebm.simulation.io import TrajectorySaver
from calphaebm.utils.constants import (
    BETA,
    EMB_DIM,
    FORCE_CAP,
    N_STEPS,
    NUM_AA,
    STEP_SIZE,
)
from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers):
    """Add simulate command parser."""
    parser = subparsers.add_parser(
        "simulate",
        description="Run Langevin simulation from PDB",
        help="Run simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdb",
        required=True,
        help="PDB ID or path",
    )

    parser.add_argument(
        "--chain",
        help="Chain ID (default: first chain)",
    )

    parser.add_argument(
        "--cache-dir",
        default="./pdb_cache",
        help="PDB cache directory",
    )

    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path",
    )

    parser.add_argument(
        "--out-dir",
        default="./runs/sim",
        help="Output directory",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=N_STEPS,
        help=f"Number of steps (default: {N_STEPS})",
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=STEP_SIZE,
        help=f"Step size (default: {STEP_SIZE})",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=BETA,
        help=f"Inverse temperature (default: {BETA})",
    )

    parser.add_argument(
        "--force-cap",
        type=float,
        default=FORCE_CAP,
        help=f"Force cap (default: {FORCE_CAP})",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Logging frequency (default: 50)",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save trajectory every N steps (default: 50)",
    )

    # Energy terms
    parser.add_argument(
        "--energy-terms",
        nargs="*",
        default=["local", "repulsion", "secondary", "packing"],
        choices=["local", "repulsion", "secondary", "packing", "all"],
        help="Energy terms to include",
    )

    parser.add_argument(
        "--repulsion-mode",
        choices=["fixed", "learned-radius"],
        default="learned-radius",
        help="Repulsion mode (default: learned-radius)",
    )

    # Lambda gates
    parser.add_argument(
        "--lambda-local",
        type=float,
        default=1.0,
        help="Local term weight (default: 1.0)",
    )

    parser.add_argument(
        "--lambda-rep",
        type=float,
        default=1.0,
        help="Repulsion term weight (default: 1.0)",
    )

    parser.add_argument(
        "--lambda-ss",
        type=float,
        default=1.0,
        help="Secondary structure term weight (default: 1.0)",
    )

    parser.add_argument(
        "--lambda-pack",
        type=float,
        default=1.0,
        help="Packing term weight (default: 1.0)",
    )

    parser.add_argument(
        "--no-dcd",
        action="store_true",
        help="Skip DCD output (only save NPY/PT)",
    )

    parser.set_defaults(func=run)


def load_chain(pdb_id: str, cache_dir: str, chain_id: str | None = None):
    """Load chain coordinates and sequence."""
    cif_path = download_cif(pdb_id, cache_dir=cache_dir)
    chains = parse_cif_ca_chains(cif_path, pdb_id.lower())

    if not chains:
        raise ValueError(f"No chains found for {pdb_id}")

    if chain_id:
        chain = next((c for c in chains if c.chain_id == chain_id), None)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found")
    else:
        chain = chains[0]
        logger.info(f"Using chain {chain.chain_id}")

    return chain.coords, chain.seq, chain.chain_id


def run(args):
    """Run simulate command."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Parse terms
    terms_set = {t.lower() for t in args.energy_terms}
    if "all" in terms_set:
        terms_set = {"local", "repulsion", "secondary", "packing"}

    # Build model
    local = LocalEnergy(
        num_aa=NUM_AA,
        emb_dim=EMB_DIM,
        hidden=(128, 128),
        use_cos_theta=True,
    ).to(device)

    rep = None
    if "repulsion" in terms_set:
        if args.repulsion_mode == "learned-radius":
            rep = RepulsionEnergyLearnedRadius().to(device)
        else:
            rep = RepulsionEnergyFixed().to(device)

    ss = None
    if "secondary" in terms_set:
        ss = SecondaryStructureEnergy(
            num_aa=NUM_AA,
            emb_dim=EMB_DIM,
        ).to(device)

    pack = None
    if "packing" in terms_set:
        pack = PackingEnergy(
            num_aa=NUM_AA,
            emb_dim=EMB_DIM,
        ).to(device)

    model = TotalEnergy(
        local=local,
        repulsion=rep,
        secondary=ss,
        packing=pack,
        gate_local=args.lambda_local,
        gate_rep=args.lambda_rep,
        gate_ss=args.lambda_ss,
        gate_pack=args.lambda_pack,
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

    # Load initial structure
    coords, seq, chain_id = load_chain(args.pdb, args.cache_dir, args.chain)

    R0 = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
    seq0 = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

    logger.info(f"Initial structure: {len(coords)} residues")

    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get residue names for topology
    residue_names = get_residue_sequence(args.pdb, args.cache_dir, chain_id)

    # Run simulation
    simulator = PyTorchSimulator(
        model,
        beta=args.beta,
        force_cap=args.force_cap,
        device=device,
    )

    from calphaebm.simulation.observers import TrajectoryObserver

    traj_obs = TrajectoryObserver(save_every=args.save_every)
    simulator.add_observer(traj_obs)

    logger.info(f"Starting simulation for {args.steps} steps")
    logger.info(f"  step_size = {args.step_size:.2e}")
    logger.info(f"  beta = {args.beta}")
    logger.info(f"  force_cap = {args.force_cap}")

    result = simulator.run(
        R0=R0,
        seq=seq0,
        n_steps=args.steps,
        step_size=args.step_size,
        log_every=args.log_every,
    )

    # Save trajectory
    logger.info("Saving trajectory...")

    saver = TrajectorySaver(out_dir, sequence=residue_names)

    for frame in result.trajectories:
        saver.append(frame)

    metadata = {
        "pdb_id": args.pdb,
        "chain_id": chain_id,
        "n_residues": len(coords),
        "n_steps": args.steps,
        "step_size": args.step_size,
        "beta": args.beta,
        "force_cap": args.force_cap,
        "terms": list(terms_set),
        "lambdas": {
            "local": args.lambda_local,
            "rep": args.lambda_rep,
            "ss": args.lambda_ss,
            "pack": args.lambda_pack,
        },
        "repulsion_mode": args.repulsion_mode,
    }

    paths = saver.save_all(metadata)

    logger.info("Saved files:")
    for fmt, path in paths.items():
        logger.info(f"  {fmt}: {path}")

    return 0
