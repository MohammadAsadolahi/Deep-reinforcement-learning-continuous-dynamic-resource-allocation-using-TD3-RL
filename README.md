<div align="center">

# Deep Reinforcement Learning for Continuous Dynamic Resource Allocation

### Twin Delayed DDPG (TD3) Applied to Real-World Multi-Resource Scheduling Under Precedence Constraints

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An end-to-end deep reinforcement learning system that learns optimal continuous resource allocation policies for complex multi-task scheduling problems — outperforming stochastic baselines in both completion speed and resource utilization.*

---

</div>

## The Problem

In real-world operations — cloud infrastructure, manufacturing, logistics — resources must be continuously allocated across hundreds of competing tasks with complex dependency graphs. Classical approaches (linear programming, heuristic schedulers) break down when:

- **Resources are continuous**, not discrete slots
- **Availability fluctuates dynamically** at every timestep
- **Precedence constraints** create cascading dependency chains across subtasks
- **Scale** reaches hundreds of concurrent tasks competing for shared resource pools

This project formulates dynamic resource allocation as a **continuous-action Markov Decision Process** and solves it using **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** — a state-of-the-art actor-critic algorithm designed for stability in continuous control.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TD3 AGENT ARCHITECTURE                   │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │              │  state  │       ACTOR NETWORK          │  │
│  │  ENVIRONMENT ├────────►│  s → 256 → 256 → 128 → a    │  │
│  │              │         │       ReLU   ReLU  Sigmoid   │  │
│  │  Resource    │◄────────┤                              │  │
│  │  Manager     │  action │  Outputs: Continuous [0, 1]  │  │
│  │              │         │  per resource per subtask    │  │
│  └──────┬───────┘         └──────────────────────────────┘  │
│         │                                                   │
│    (s, a, r, s')          ┌──────────────────────────────┐  │
│         │                 │     TWIN CRITIC NETWORKS     │  │
│         ▼                 │                              │  │
│  ┌──────────────┐         │  Q1: (s,a) → 256 → 256 → 1  │  │
│  │   REPLAY     │────────►│  Q2: (s,a) → 256 → 256 → 1  │  │
│  │   BUFFER     │  batch  │                              │  │
│  │   (1M)       │         │  min(Q1, Q2) prevents        │  │
│  └──────────────┘         │  overestimation bias         │  │
│                           └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Why TD3?

TD3 ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)) extends DDPG with three critical innovations that make it suitable for high-dimensional resource allocation:

| Technique | Purpose | Implementation |
|:--|:--|:--|
| **Twin Critics** | Eliminates Q-value overestimation by taking $\min(Q_1, Q_2)$ | Two parallel critic networks with independent parameters |
| **Delayed Policy Updates** | Stabilizes training by updating the actor less frequently than the critic | Actor updates every 2 critic updates |
| **Target Policy Smoothing** | Regularizes the value estimate by adding clipped noise to target actions | $\tilde{a} = \pi_{\theta'}(s') + \text{clip}(\epsilon, -c, c)$ where $\epsilon \sim \mathcal{N}(0, \sigma)$ |

---

## Environment Design

The custom `Resource_Manager` environment models real-world resource scheduling:

### State Space

$$s_t = [\; r_t^{\text{avail}}, \; d_1, \; d_2, \; \dots, \; d_n \;]$$

| Component | Description |
|:--|:--|
| $r_t^{\text{avail}}$ | Available resource quantity at timestep $t$ (stochastic) |
| $d_i$ | Remaining demand for subtask $i$ |

### Action Space

$$a_t \in [0, 1]^n$$

Continuous allocation fractions for each of $n$ subtasks, normalized via softmax against available resources. The agent learns *how much* of each resource to allocate — not a binary assign/skip decision.

### Reward Signal

$$R_t = \begin{cases} +1000 & \text{if all demands satisfied (terminal)} \\ -1 & \text{otherwise (step penalty)} \end{cases}$$

The sparse terminal reward combined with dense step penalties drives the agent toward completing all tasks in minimum time.

---

## Two-Stage Validation

### Stage 1 — Proof of Concept (6 subtasks)

A compact environment validates the algorithm's core mechanics:

- **6 continuous actions** over synthetic resource demands
- **2,000 training episodes** at 2,000 max steps each
- Manual dependency structures (`order1`, `order2`) of increasing complexity
- Confirms the TD3 agent learns meaningful allocation policies vs. random baseline

### Stage 2 — Real-World Scale (644 subtasks)

The algorithm is then validated against production-scale scheduling data:

- **644 subtasks** organized across multi-level task hierarchies
- **4 resource types** (R1–R4) with independent availability pools
- **Precedence constraints** loaded from real CSV datasets defining which subtasks must complete before others can begin
- **5,000 training episodes** with deeper 4-layer networks (256→256→128)

**Data Pipeline:**

```
task_subtask.csv       →  Task-to-subtask decomposition
sub_task_successor.csv →  Precedence constraint graph
duration.csv           →  Subtask execution durations
resources.csv          →  Resource pool capacities (R1–R4)
requirements.csv       →  Per-subtask resource consumption matrix (644 × 4)
```

---

## Results

### Trained Policy vs. Random Baseline

| Metric | Trained TD3 | Random Policy |
|:--|:--|:--|
| **Task Completion Speed** | Significantly faster convergence to 100% | Slow, erratic progress |
| **Resource Utilization** | Consistently high (near-optimal) | Low and volatile |
| **Subtask Completion Curve** | Smooth, monotonic increase | Noisy, plateaus frequently |

The trained agent maintains **sustained high resource utilization** across all four resource types, while the random baseline exhibits wasteful allocation with utilization spikes and dead periods.

### Training Dynamics

- **Reward curve** shows clear learning signal after the 10,000-step exploration phase
- **Cumulative average reward** demonstrates stable policy improvement across 5,000 episodes
- No catastrophic forgetting observed — the delayed updates and twin critics provide training stability at scale

---

## Project Structure

```
.
├── Actor.py                              # Actor network (policy)
├── Critic.py                             # Twin critic networks (Q1, Q2)
├── TD3.py                                # TD3 algorithm (train, select_action, save/load)
├── Replay_Buffer.py                      # Experience replay memory (1M transitions)
├── Utils.py                              # Soft/hard target updates, softmax
├── imports.py                            # Shared dependencies
├── Dynamic_resource_allocation_TD3.ipynb  # Stage 1: Proof-of-concept (6 subtasks)
└── RL alloc.ipynb                        # Stage 2: Full-scale evaluation (644 subtasks)
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|:--|:--|:--|
| Discount $\gamma$ | 0.99 | Long-horizon task — future rewards matter |
| Soft update $\tau$ | 0.005 | Slow target tracking for stability |
| Policy noise $\sigma$ | 0.2 | Smooths value estimates around actions |
| Noise clip $c$ | 0.5 | Prevents extreme target actions |
| Actor update frequency | Every 2 critic steps | Core TD3 delayed update mechanism |
| Learning rate | 3×10⁻⁴ (Adam) | Standard for continuous control |
| Replay buffer | 10⁶ transitions | Sufficient decorrelation for off-policy learning |
| Batch size | 100 | Balances gradient noise and compute |
| Exploration | $\mathcal{N}(0, 0.1)$ for first 10K steps | Ensures diverse initial experience collection |

---

## Quick Start

### Prerequisites

```bash
pip install numpy torch
```

### Train (Proof of Concept)

Run all cells in `Dynamic_resource_allocation_TD3.ipynb` to train the 6-subtask environment and compare trained vs. random allocation.

### Train (Full Scale)

1. Place your scheduling CSV files (`task_subtask.csv`, `sub_task_successor.csv`, `duration.csv`, `resources.csv`, `requirements.csv`) in the project root
2. Run all cells in `RL alloc.ipynb`
3. Trained weights are saved as `RL_alloc_actor`, `RL_alloc_critic`, etc.

### Evaluate

```python
from TD3 import TD3

policy = TD3(state_dim=645, action_dim=644, max_action=1.0)
policy.load("RL_alloc")

state = env.reset()
action = policy.select_action(state)
```

---

## Mathematical Formulation

**Critic Loss (both $Q_1$ and $Q_2$):**

$$\mathcal{L}_{\text{critic}} = \frac{1}{|B|} \sum_{(s,a,r,s') \in B} \left[ \left( Q_i(s,a) - y \right)^2 \right]$$

where the target is:

$$y = r + \gamma \cdot \min_{i=1,2} Q_{\theta'_i}(s', \tilde{a}'), \quad \tilde{a}' = \pi_{\phi'}(s') + \text{clip}(\epsilon, -c, c)$$

**Actor Loss (delayed, every 2 steps):**

$$\mathcal{L}_{\text{actor}} = -\frac{1}{|B|} \sum_{s \in B} Q_1(s, \pi_\phi(s))$$

**Target Network Update:**

$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

---

## References

- Fujimoto, S., Hoof, H., & Meger, D. (2018). [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477). *ICML 2018.*
- Lillicrap, T. P., et al. (2016). [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971). *ICLR 2016.*

---

<div align="center">

**Built with PyTorch** · **Designed for Production-Scale Scheduling**

</div>
