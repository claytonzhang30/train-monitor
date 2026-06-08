"""Training metric tracker with running statistics."""

import time
import threading
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class RunningStats:
    """Online computation of mean, variance, min, max using Welford's algorithm."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    ema: float = 0.0
    ema_alpha: float = 0.1

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.ema = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "ema": round(self.ema, 6),
        }


@dataclass
class MetricEntry:
    """A single metric data point."""
    step: int
    value: float
    timestamp: float
    phase: str  # "train" or "val"


class TrainingTracker:
    """Track and compute running statistics for training metrics.

    Thread-safe for concurrent logging from multiple threads (e.g., GPU monitor
    thread and training thread).

    Args:
        experiment_name: Name for this training run.
        window_size: Number of recent entries to keep in memory.
        ema_alpha: Exponential moving average smoothing factor.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        window_size: int = 10000,
        ema_alpha: float = 0.1,
    ):
        self.experiment_name = experiment_name
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self._lock = threading.Lock()

        # metric_name → deque of MetricEntry
        self._history: Dict[str, Deque[MetricEntry]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # metric_name → RunningStats
        self._stats: Dict[str, RunningStats] = defaultdict(
            lambda: RunningStats(ema_alpha=ema_alpha)
        )
        # Separate stats per phase
        self._phase_stats: Dict[Tuple[str, str], RunningStats] = defaultdict(
            lambda: RunningStats(ema_alpha=ema_alpha)
        )

        self._start_time = time.time()
        self._last_step = 0
        self._step_times: Deque[float] = deque(maxlen=100)

    def log(
        self,
        step: int,
        metrics: Dict[str, float],
        phase: str = "train",
    ) -> None:
        """Log metrics for a training step.

        Args:
            step: Global training step number.
            metrics: Dict of metric_name → value.
            phase: "train" or "val" (or custom phase name).
        """
        now = time.time()
        with self._lock:
            # Track step timing
            if self._last_step > 0 and step > self._last_step:
                steps_delta = step - self._last_step
                time_per_step = (now - self._get_last_timestamp()) / max(steps_delta, 1)
                for _ in range(min(steps_delta, 10)):
                    self._step_times.append(time_per_step)

            self._last_step = max(self._last_step, step)

            for name, value in metrics.items():
                entry = MetricEntry(step=step, value=value, timestamp=now, phase=phase)
                self._history[name].append(entry)
                self._stats[name].update(value)
                self._phase_stats[(name, phase)].update(value)

    def _get_last_timestamp(self) -> float:
        """Get the timestamp of the most recent entry across all metrics."""
        latest = self._start_time
        for hist in self._history.values():
            if hist:
                latest = max(latest, hist[-1].timestamp)
        return latest

    def get_stats(self, metric_name: str, phase: Optional[str] = None) -> Optional[dict]:
        """Get running statistics for a metric.

        Args:
            metric_name: Name of the metric.
            phase: If specified, return stats only for this phase.

        Returns:
            Dict with count, mean, std, min, max, ema — or None if metric not found.
        """
        with self._lock:
            if phase:
                key = (metric_name, phase)
                stats = self._phase_stats.get(key)
            else:
                stats = self._stats.get(metric_name)
            return stats.to_dict() if stats and stats.count > 0 else None

    def get_history(
        self,
        metric_name: str,
        last_n: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> List[dict]:
        """Get metric history as list of dicts.

        Args:
            metric_name: Name of the metric.
            last_n: Return only the last N entries. None = all.
            phase: Filter by phase.

        Returns:
            List of {step, value, timestamp, phase}.
        """
        with self._lock:
            entries = list(self._history.get(metric_name, []))
            if phase:
                entries = [e for e in entries if e.phase == phase]
            if last_n:
                entries = entries[-last_n:]
            return [
                {"step": e.step, "value": e.value, "timestamp": e.timestamp, "phase": e.phase}
                for e in entries
            ]

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        with self::