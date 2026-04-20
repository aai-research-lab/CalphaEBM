"""Self-consistent training defaults — thin wrapper around defaults.py.

Imports SC section from the master defaults file (#7).
Kept as a separate module for backward compatibility —
existing code imports SC_DEFAULTS from here.

Usage:
    from calphaebm.training.sc_defaults import SC_DEFAULTS
    margin = SC_DEFAULTS["sc_margin"]
"""

from calphaebm.defaults import SC as SC_DEFAULTS

# CLI arg name → __init__ param name (only where they differ)
CLI_TO_INIT = {
    "n_workers": "collect_n_workers",
    "lambda_funnel": "lambda_elt",
    "lambda_native_depth": "lambda_depth",
    "target_native_depth": "target_depth",
    "sc_eval_steps": "eval_steps",
    "sc_eval_beta": "eval_beta",
    "sc_eval_proteins": "eval_proteins",
}


def get_default(init_name: str):
    """Get default value by __init__ parameter name."""
    return SC_DEFAULTS[init_name]
