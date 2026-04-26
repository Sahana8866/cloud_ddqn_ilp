# Hybrid DDQN-ILP Intelligent Offloading with Federated Learning

**Achievement:** 20-30% latency reduction | ~85% model accuracy

---

## Architecture

3-tier system: IoT Devices → Edge Nodes → Cloud Server

| Component | Role |
|-----------|------|
| **Cloud** | DDQN policy, FL aggregation, ILP fallback |
| **Edge** | Local FL training, stress monitoring |
| **Device** | Data collection, local training |

---

## How It Works

1. **Decision:** DDQN selects edge for each device (ILP fallback during exploration)
2. **Offload:** Devices send data to assigned edges
3. **FL Training:** Edges train locally → Cloud aggregates globally
4. **Reward:** Based on latency, energy, CPU usage

---

## Results

- 20-30% latency reduction
- ~85% model accuracy maintained
- 50 FL rounds with dynamic stress scenarios

---

## Tech Stack

Python, PyTorch, Flask, psutil, Sockets

---

## Files

| File | Purpose |
|------|---------|
| `cloudcore.py` | Cloud server (DDQN + FL aggregation) |
| `edge.py` | Edge node (local training) |
| `device.py` | IoT device (data offloading) |

---

## Run

```bash
pip install torch torchvision numpy psutil flask
python cloudcore.py   # Terminal 1
python edge.py        # Terminal 2
python device.py 0    # Terminal 3