# src/calphaebm/cli/commands/evaluate.py

"""Evaluation command."""

import argparse
from pathlib import Path

from calphaebm.evaluation.plotting import plot_all
from calphaebm.evaluation.reporting import TrajectoryEvaluator
from calphaebm.utils.logging import get_logger

logger = get_logger()


def add_parser(subparsers):
    """Add evaluate command parser."""
    parser = subparsers.add_parser(
        "evaluate",
        description="Evaluate a trajectory",
        help="Evaluate trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--traj",
        required=True,
        help="Trajectory directory",
    )

    parser.add_argument(
        "--ref-pt",
        help="Reference .pt file (default: snapshot_0000.pt)",
    )

    parser.add_argument(
        "--ref-xyz",
        help="Reference .xyz file (alternative to --ref-pt)",
    )

    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=8.0,
        help="Contact cutoff in A (default: 8.0)",
    )

    parser.add_argument(
        "--exclude",
        type=int,
        default=2,
        help="Sequence exclusion (default: 2)",
    )

    parser.add_argument(
        "--rdf-rmax",
        type=float,
        default=20.0,
        help="RDF max distance in A (default: 20.0)",
    )

    parser.add_argument(
        "--rdf-dr",
        type=float,
        default=0.25,
        help="RDF bin width in A (default: 0.25)",
    )

    parser.add_argument(
        "--q-smooth-beta",
        type=float,
        default=5.0,
        help="Q_smooth beta parameter (default: 5.0)",
    )

    parser.add_argument(
        "--q-smooth-lambda",
        type=float,
        default=1.8,
        help="Q_smooth lambda parameter (default: 1.8)",
    )

    parser.add_argument(
        "--clash-threshold",
        type=float,
        default=3.8,
        help="Clash threshold in A (default: 3.8)",
    )

    parser.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="Frames to discard for burn-in (default: 0)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    parser.set_defaults(func=run)


def run(args):
    """Run evaluate command."""
    # Determine reference path
    ref_path = None
    if args.ref_xyz:
        ref_path = args.ref_xyz
    elif args.ref_pt:
        ref_path = args.ref_pt
    else:
        ref_path = str(Path(args.traj) / "snapshot_0000.pt")

    # Create evaluator
    evaluator = TrajectoryEvaluator(
        contact_cutoff=args.contact_cutoff,
        exclude=args.exclude,
        rdf_rmax=args.rdf_rmax,
        rdf_dr=args.rdf_dr,
        q_smooth_beta=args.q_smooth_beta,
        q_smooth_lambda=args.q_smooth_lambda,
        clash_threshold=args.clash_threshold,
    )

    # Run evaluation
    logger.info(f"Evaluating trajectory: {args.traj}")
    logger.info(f"Reference: {ref_path}")

    report = evaluator.evaluate_from_dir(
        args.traj,
        ref_path=ref_path,
        burnin=args.burnin,
    )

    # Print summary
    print("\n" + report.summary())

    # Save results
    out_dir = Path(args.traj) / "eval"
    report.save(out_dir)
    logger.info(f"Saved results to {out_dir}")

    # Generate plots
    if not args.no_plots:
        logger.info("Generating plots...")
        plot_all(report, out_dir)
        logger.info(f"Saved plots to {out_dir}")

    return 0
