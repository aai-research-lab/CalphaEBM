# src/calphaebm/cli/commands/train.py

"""Training command."""

import argparse
import os
import sys
from typing import List, Set

import torch
from torch.utils.data import DataLoader

from calphaebm.utils.logging import get_logger, ProgressBar
from calphaebm.utils.seed import seed_all
from calphaebm.utils.constants import (
    NUM_AA, EMB_DIM, DSM_SIGMA, LEARNING_RATE,
)
from calphaebm.models.local_terms import LocalEnergy
from calphaebm.models.repulsion import (
    RepulsionEnergyFixed,
    RepulsionEnergyLearnedRadius,
)
from calphaebm.models.cross_terms import SecondaryStructureEnergy
from calphaebm.models.packing import PackingEnergy
from calphaebm.models.energy import TotalEnergy
from calphaebm.data.pdb_dataset import PDBSegmentDataset
from calphaebm.training.losses.dsm import dsm_cartesian_loss
from calphaebm.training.losses.contrastive import contrastive_logistic_loss
from calphaebm.training.freeze import freeze_module, unfreeze_module
from calphaebm.geometry.internal import bond_angles, torsions
from calphaebm.simulation.backends.pytorch import langevin_sample

logger = get_logger()


def add_parser(subparsers):
    """Add train command parser."""
    parser = subparsers.add_parser(
        "train",
        description="Train energy model with phased training",
        help="Train energy model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument(
        "--phase",
        choices=["local", "repulsion", "secondary", "packing", "all"],
        required=True,
        help="Training phase",
    )
    
    parser.add_argument(
        "--pdb",
        nargs="+",
        required=True,
        help="PDB IDs or file containing IDs (one per line)",
    )
    
    # Data options
    parser.add_argument(
        "--cache-dir",
        default="./pdb_cache",
        help="Directory for PDB cache",
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
        default=5000,
        help="Maximum number of segments to load",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    
    # Training options
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of training steps (default: 2000)",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    
    parser.add_argument(
        "--sigma",
        type=float,
        default=DSM_SIGMA,
        help=f"DSM noise sigma in A (default: {DSM_SIGMA})",
    )
    
    # Checkpoint options
    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    
    parser.add_argument(
        "--ckpt-prefix",
        default="run1",
        help="Experiment prefix (default: run1)",
    )
    
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    
    parser.add_argument(
        "--resume",
        help="Checkpoint path to resume from, or 'auto' for latest",
    )
    
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load only model weights, not optimizer state",
    )
    
    # Energy terms
    parser.add_argument(
        "--energy-terms",
        nargs="*",
        default=["local"],
        choices=["local", "repulsion", "secondary", "packing", "all"],
        help="Energy terms to include (default: local)",
    )
    
    parser.add_argument(
        "--freeze",
        nargs="*",
        default=[],
        choices=["local", "repulsion", "secondary", "packing"],
        help="Terms to freeze",
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
    
    # Repulsion options
    parser.add_argument(
        "--repulsion-mode",
        choices=["fixed", "learned-radius"],
        default="learned-radius",
        help="Repulsion mode (default: learned-radius)",
    )
    
    parser.add_argument(
        "--rep-neg-shift",
        type=float,
        default=0.6,
        help="Negative shift for learned repulsion (default: 0.6)",
    )
    
    parser.add_argument(
        "--rep-radius-reg",
        type=float,
        default=1e-3,
        help="Radius regularization strength (default: 1e-3)",
    )
    
    # Packing options
    parser.add_argument(
        "--pack-noise",
        type=float,
        default=1.0,
        help="Noise for negative generation (default: 1.0)",
    )
    
    parser.add_argument(
        "--neg-langevin-steps",
        type=int,
        default=20,
        help="Langevin steps for negative generation (default: 20)",
    )
    
    parser.add_argument(
        "--neg-step-size",
        type=float,
        default=2e-4,
        help="Step size for negative Langevin (default: 2e-4)",
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


def build_model(terms_set: Set[str], device, args):
    """Build energy model."""
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
    
    return model


def run(args):
    """Run train command."""
    # Set seed
    seed_all(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Parse PDB IDs
    pdb_ids = parse_pdb_arg(args.pdb)
    if not pdb_ids:
        logger.error("No valid PDB IDs found")
        return 1
    
    logger.info(f"Using {len(pdb_ids)} PDB entries")
    
    # Parse terms
    terms_set = {t.lower() for t in args.energy_terms}
    if "all" in terms_set:
        terms_set = {"local", "repulsion", "secondary", "packing"}
    terms_set.add("local")  # Always include local
    
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
    
    logger.info(f"Dataset: {len(ds)} segments")
    
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    # Build model
    model = build_model(terms_set, device, args)
    
    # Apply freezing
    freeze_set = {t.lower() for t in args.freeze}
    for term in freeze_set:
        if hasattr(model, term):
            freeze_module(getattr(model, term))
            logger.info(f"Froze {term}")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) > 0:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        logger.warning("No trainable parameters in this phase")
        optimizer = None
    
    # Resume if requested
    start_step = 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path.lower() == "auto":
            # Find latest checkpoint
            phase_dir = os.path.join(args.ckpt_dir, args.ckpt_prefix, args.phase)
            if os.path.exists(phase_dir):
                ckpts = [f for f in os.listdir(phase_dir) if f.startswith("step")]
                if ckpts:
                    steps = [int(f.replace("step", "").replace(".pt", "")) for f in ckpts]
                    latest_idx = max(range(len(steps)), key=lambda i: steps[i])
                    ckpt_path = os.path.join(phase_dir, ckpts[latest_idx])
        
        if os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            if not args.resume_model_only and optimizer and "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = ckpt.get("step", 0)
    
    # Training loop
    model.train()
    data_iter = iter(dl)
    
    progress = ProgressBar(args.steps, prefix=f"Phase {args.phase}")
    
    for step in range(start_step + 1, args.steps + 1):
        # Get batch
        try:
            coords, seq, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            coords, seq, _, _ = next(data_iter)
        
        coords = coords.to(device)
        seq = seq.to(device)
        
        # Compute loss based on phase
        if args.phase == "local":
            loss = dsm_cartesian_loss(model, coords, seq, sigma=args.sigma)
        
        elif args.phase == "secondary":
            if model.secondary is None:
                logger.error("Secondary module not enabled")
                return 1
            
            theta = bond_angles(coords)
            phi = torsions(coords)
            
            # Shuffle phi to break correlations
            B, nphi = phi.shape
            perm = torch.stack([torch.randperm(nphi, device=device) for _ in range(B)], dim=0)
            phi_neg = torch.gather(phi, 1, perm)
            
            E_pos = model.secondary.energy_from_thetaphi(theta, phi, seq)
            E_neg = model.secondary.energy_from_thetaphi(theta, phi_neg, seq)
            
            loss = contrastive_logistic_loss(E_pos, E_neg)
        
        elif args.phase == "packing":
            if model.packing is None:
                logger.error("Packing module not enabled")
                return 1
            
            # Generate negatives with noise + short Langevin
            R_noisy = coords + args.pack_noise * torch.randn_like(coords)
            
            snaps = langevin_sample(
                model,
                R0=R_noisy,
                seq=seq,
                n_steps=args.neg_langevin_steps,
                step_size=args.neg_step_size,
                force_cap=50.0,
                log_every=1000,  # Disable logging
            )
            R_neg = snaps[-1].to(device)
            
            E_pos = model.packing(coords, seq)
            E_neg = model.packing(R_neg, seq)
            
            loss = contrastive_logistic_loss(E_pos, E_neg)
        
        elif args.phase == "repulsion":
            if model.repulsion is None:
                logger.error("Repulsion module not enabled")
                return 1
            
            if args.repulsion_mode == "fixed":
                # Calibration mode
                from calphaebm.utils.neighbors import pairwise_distances
                
                with torch.no_grad():
                    # Compute minimum nonbonded distance
                    B, L, _ = coords.shape
                    D = pairwise_distances(coords)
                    
                    # Mask bonded pairs
                    for i in range(L):
                        D[:, i, max(0, i-2):min(L, i+3)] = float("inf")
                    
                    min_dist = D.amin(dim=(1, 2)).median().item()
                    
                    # Adjust gate based on min distance
                    current = model.gate_rep.item()
                    target_min = 3.6
                    
                    if min_dist < target_min:
                        new_gate = current * 1.1
                    else:
                        new_gate = current * 0.9
                    
                    new_gate = max(0.1, min(100.0, new_gate))
                    model.set_gates(gate_rep=new_gate)
                    
                    # Dummy loss
                    loss = torch.tensor(min_dist, device=device, requires_grad=False)
                
                logger.info(f"Step {step}: min_dist={min_dist:.3f}, gate_rep={new_gate:.3f}")
                
            else:
                # Learned-radius training
                repmod = model.repulsion
                from calphaebm.utils.neighbors import topk_nonbonded_pairs
                
                # Get positive pairs
                r_pos, j = topk_nonbonded_pairs(coords, K=repmod.K, exclude=repmod.exclude)
                B, L, K = j.shape
                
                seq_i = seq.unsqueeze(-1).expand(-1, -1, K)
                seq_j = torch.gather(seq.unsqueeze(-1).expand(-1, -1, K), 1, j)
                
                # Negative distances (shrunk)
                r_neg = torch.clamp(r_pos - args.rep_neg_shift, min=0.5)
                
                # Compute energies
                r0 = repmod.pair_r0(seq_i, seq_j)
                x_pos = (r0 - r_pos) / (repmod.delta + 1e-12)
                x_neg = (r0 - r_neg) / (repmod.delta + 1e-12)
                
                e_pos = repmod.wall_scale * torch.nn.functional.softplus(x_pos).mean()
                e_neg = repmod.wall_scale * torch.nn.functional.softplus(x_neg).mean()
                
                loss = torch.log1p(torch.exp(e_pos - e_neg))
                
                # Radius regularization
                rho = repmod.rho(seq)
                loss = loss + args.rep_radius_reg * ((rho - repmod.rho_base) ** 2).mean()
        
        else:
            logger.error(f"Unknown phase: {args.phase}")
            return 1
        
        # Optimizer step
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
        
        # Logging
        if step == 1 or step % 50 == 0:
            logger.info(f"Step {step:6d}/{args.steps} | loss = {loss.item():.6f}")
        
        # Save checkpoint
        if step % args.ckpt_every == 0:
            phase_dir = os.path.join(args.ckpt_dir, args.ckpt_prefix, args.phase)
            os.makedirs(phase_dir, exist_ok=True)
            
            ckpt_path = os.path.join(phase_dir, f"step{step:06d}.pt")
            
            payload = {
                "step": step,
                "phase": args.phase,
                "loss": loss.item(),
                "model_state": model.state_dict(),
                "gates": model.get_gates(),
            }
            if optimizer is not None:
                payload["optimizer_state"] = optimizer.state_dict()
            
            torch.save(payload, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
        
        progress.update(1)
    
    # Save final checkpoint
    phase_dir = os.path.join(args.ckpt_dir, args.ckpt_prefix, args.phase)
    os.makedirs(phase_dir, exist_ok=True)
    
    final_path = os.path.join(phase_dir, f"step{args.steps:06d}.pt")
    payload = {
        "step": args.steps,
        "phase": args.phase,
        "loss": loss.item() if 'loss' in locals() else 0.0,
        "model_state": model.state_dict(),
        "gates": model.get_gates(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    
    torch.save(payload, final_path)
    logger.info(f"Training complete: {final_path}")
    
    return 0