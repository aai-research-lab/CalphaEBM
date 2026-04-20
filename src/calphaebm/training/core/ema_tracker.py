# src/calphaebm/training/core/ema_tracker.py
"""Exponential moving average tracker for training metrics.

Single-batch diagnostics are noisy — Z-score can swing ±1 between batches.
EMA provides a stable trend that's meaningful for monitoring convergence.

Usage:
    tracker = EMATracker(alpha=0.02)
    # Every training step:
    tracker.update(dsm=2.1, z_score=1.5, balance=50.0)
    # At diagnostic time:
    ema = tracker.get()
    print(f"Z̄={ema['z_score']:.2f}")  # smooth trend
"""

from __future__ import annotations

from typing import Dict, Optional


class EMATracker:
    """Exponential moving average tracker for named metrics.

    EMA formula: ema_new = α * value + (1-α) * ema_old
    Default α=0.02 gives ~50-step half-life (ln2/α ≈ 35 steps to 50% weight).

    Args:
        alpha: Smoothing factor. Smaller = smoother. 0.02 is good for
               metrics logged every step with diagnostics every 200 steps.
    """

    def __init__(self, alpha: float = 0.02):
        self.alpha = float(alpha)
        self._ema: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def update(self, **kwargs: float) -> None:
        """Update EMA for one or more metrics.

        Args:
            **kwargs: metric_name=value pairs. None values are skipped.
        """
        for name, value in kwargs.items():
            if value is None:
                continue
            value = float(value)
            if name not in self._ema:
                # First observation: initialize to exact value (no bias)
                self._ema[name] = value
                self._count[name] = 1
            else:
                self._ema[name] = self.alpha * value + (1.0 - self.alpha) * self._ema[name]
                self._count[name] += 1

    def get(self, name: Optional[str] = None) -> Dict[str, float] | float | None:
        """Get current EMA value(s).

        Args:
            name: If provided, return single float (or None if not tracked).
                  If None, return dict of all tracked metrics.
        """
        if name is not None:
            return self._ema.get(name)
        return dict(self._ema)

    def get_count(self, name: str) -> int:
        """Number of observations for a metric."""
        return self._count.get(name, 0)

    def has(self, name: str) -> bool:
        """Check if a metric is being tracked."""
        return name in self._ema

    def format(self, metrics: list[str] | None = None, precision: int = 3) -> str:
        """Format EMA values as a compact string for logging.

        Args:
            metrics: List of metric names to include. None = all.
            precision: Decimal places.

        Returns:
            String like "dsm=2.156  Z̄=1.52  bal=32.1"
        """
        names = metrics or sorted(self._ema.keys())
        parts = []
        for name in names:
            if name in self._ema:
                parts.append(f"{name}={self._ema[name]:.{precision}f}")
        return "  ".join(parts)

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._ema.clear()
        self._count.clear()
