"""train-monitor: Real-time training dashboard."""

__version__ = "0.1.0"

from monitor.tracker import TrainingTracker
from monitor.gpu import GPUMonitor
from monitor.alerts import AlertManager

__all__ = ["TrainingTracker", "GPUMonitor", "AlertManager"]
