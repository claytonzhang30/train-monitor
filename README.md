# train-monitor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Real-time training dashboard for deep learning workflows. Monitor loss curves, GPU utilization, gradient statistics, and set threshold alerts — all from a browser.

## Features

- **Loss Tracking** — Log and visualize training/validation loss with running statistics (EMA, min, max)
- **GPU Monitoring** — Real-time GPU utilization, memory usage, temperature via nvidia-smi or rocm-smi
- **Gradient Stats** — Track gradient norms, detect exploding/vanishing gradients
- **Alert System** — Configurable threshold alerts for loss spikes, GPU OOM, temperature warnings
- **Live Dashboard** — Flask-powered web UI with auto-updating charts (Chart.js)
- **Framework Agnostic** — Works with PyTorch, TensorFlow, JAX, or any framework

## Quick Start

```bash
pip install -e .
```

### In your training script

```python
from monitor.tracker import TrainingTracker
from monitor.alerts import AlertManager

tracker = TrainingTracker(experiment_name="llama-finetune")
alerts = AlertManager(tracker)

# Log metrics each step
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    tracker.log(step=step, metrics={"loss": loss, "lr": scheduler.get_last_lr()[0]})

    if step % 100 == 0:
        val_loss = evaluate(val_loader)
        tracker.log(step=step, metrics={"val_loss": val_loss}, phase="val")

    # Check alerts
    triggered = alerts.check()
    for alert in triggered:
        print(f"⚠️  {alert}")
```

### Launch the dashboard

```python
from monitor.dashboard import create_app

app = create_app(tracker)
app.run(host="0.0.0.0", port=5000)
```

Or from the command line:

```bash
train-monitor --experiment llama-finetune --port 5000
```

Open `http://localhost:5000` in your browser.

## Configuration

```python
from monitor.alerts import AlertManager

alerts = AlertManager(
    tracker=tracker,
    rules={
        "loss_spike": {"metric": "loss", "threshold": 2.0, "mode": "relative_increase"},
        "gpu_memory": {"threshold": 0.95, "mode": "fraction"},
        "gpu_temp": {"threshold": 85, "mode": "celsius"},
        "grad_explosion": {"metric": "grad_norm", "threshold": 100.0, "mode": "absolute"},
    },
    cooldown_seconds=60,
)
```

## Architecture

```
TrainingTracker → in-memory metrics store + stats
     ↓
GPUMonitor → polls nvidia-smi / rocm-smi every N seconds
     ↓
AlertManager → evaluates rules against metrics + GPU state
     ↓
Flask Dashboard → serves live data via REST API + Chart.js frontend
```

## License

MIT License — see [LICENSE](LICENSE) for details.

<!-- history: 2026-05-18 -->

<!-- history: 2026-05-24 -->
