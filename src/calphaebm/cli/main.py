# src/calphaebm/cli/main.py

"""Main CLI entry point."""

import argparse
import sys
from typing import List

from calphaebm import __version__
from calphaebm.utils.logging import setup_logger

logger = setup_logger("calphaebm.cli")


def main(args: List[str] = None) -> int:
    """Main CLI entry point."""
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        description="CalphaEBM: Cα Energy-Based Model for Protein Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"CalphaEBM {__version__}",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Valid commands",
        help="Additional help",
    )
    
    # Import command modules
    from calphaebm.cli.commands.train import add_parser as add_train_parser
    from calphaebm.cli.commands.simulate import add_parser as add_simulate_parser
    from calphaebm.cli.commands.balance import add_parser as add_balance_parser
    from calphaebm.cli.commands.evaluate import add_parser as add_evaluate_parser
    from calphaebm.cli.commands.build_dataset import add_parser as add_build_dataset_parser
    
    add_train_parser(subparsers)
    add_simulate_parser(subparsers)
    add_balance_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_build_dataset_parser(subparsers)
    
    # Parse arguments
    parsed = parser.parse_args(args)
    
    # Set logging level
    if parsed.verbose:
        setup_logger(level="DEBUG")
    
    # Run command
    if parsed.command == "train":
        from calphaebm.cli.commands.train import run
        return run(parsed)
    elif parsed.command == "simulate":
        from calphaebm.cli.commands.simulate import run
        return run(parsed)
    elif parsed.command == "balance":
        from calphaebm.cli.commands.balance import run
        return run(parsed)
    elif parsed.command == "evaluate":
        from calphaebm.cli.commands.evaluate import run
        return run(parsed)
    elif parsed.command == "build-dataset":
        from calphaebm.cli.commands.build_dataset import run
        return run(parsed)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())