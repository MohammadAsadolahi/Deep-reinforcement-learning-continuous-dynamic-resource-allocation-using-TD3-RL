<div align="center">

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/%E2%9A%A1-TD3%20Resource%20Allocator-00d4aa?style=for-the-badge&labelColor=0d1117">
  <img alt="TD3 Resource Allocator" src="https://img.shields.io/badge/%E2%9A%A1-TD3%20Resource%20Allocator-00d4aa?style=for-the-badge&labelColor=0d1117">
</picture>

# Deep Reinforcement Learning for<br/>Continuous Dynamic Resource Allocation

### An RL agent that learns to distribute continuous resources across competing subtasks with dependency constraints — comparing learned policies against random baselines.

<br/>

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/TD3-arXiv%3A1802.09477-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/1802.09477)

<br/>

**Author:** [Mohammad Asadolahi](https://github.com/MohammadAsadolahi) · Senior Agentic AI Engineer · Focus: Agentic AI Architectures In The Wild

<br/>

</div>

---

<br/>

## Why This Exists

Resource allocation is one of the oldest problems in operations research. When you have hundreds of subtasks competing for multiple resource types, each with **stochastic availability** and **cascading precedence constraints**, finding good allocation policies becomes challenging.

This project takes a different approach: **treat the entire allocation problem as a continuous-action MDP and let a deep RL agent learn the policy from scratch.**

The agent doesn't know what "resource allocation" means. It only knows states, actions, and rewards. The project validates this approach at two scales — a 6-subtask proof of concept and a 644-subtask production-scale experiment — and compares the trained agent's performance against a random (untrained) baseline.

<br/>

---

<br/>

## How It Works

<br/>

### The Core Insight

Instead of discretizing resources into slots or writing rules for priority queues, we let the agent output a **continuous allocation vector** $a_t \in [0, 1]^n$ at every timestep. This vector is softmax-normalized against available resources, producing a smooth, differentiable allocation policy that can be optimized end-to-end via gradient descent.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │          TD3 AGENT ARCHITECTURE (Production Scale)          │
                    │                                                             │
                    │  ┌──────────────┐         ┌──────────────────────────────┐  │
                    │  │              │  state   │        ACTOR NETWORK        │  │
                    │  │  Resource    ├─────────►│                             │  │
                    │  │  Manager     │          │  s → 256 → 256 → 128 → a   │  │
                    │  │              │◄─────────┤       ReLU   ReLU  Sigmoid  │  │
                    │  │  644 subtasks│  action  │                             │  │
                    │  │  4 resources │          │  Output: [0,1]^644          │  │
                    │  └──────┬───────┘          └──────────────────────────────┘  │
                    │         │                                                    │
                    │    (s, a, r, s')           ┌──────────────────────────────┐  │
                    │         │                  │    TWIN CRITIC NETWORKS      │  │
                    │         ▼                  │                              │  │
                    │  ┌──────────────┐          │  Q₁: (s,a) → 256→256→128→1  │  │
                    │  │   REPLAY     │─────────►│  Q₂: (s,a) → 256→256→128→1  │  │
                    │  │   BUFFER     │  batch   │                              │  │
                    │  │   (1M)       │          │  min(Q₁, Q₂) eliminates     │  │
                    │  └──────────────┘          │  overestimation bias         │  │
                    │                            └──────────────────────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
```

> **Note:** The diagram above shows the production-scale (Stage 2) architecture from `RL alloc.ipynb`. The proof-of-concept (Stage 1) uses a 3-layer network (256 → 256 → out). The standalone module files (`Actor.py`, `Critic.py`) provide a base implementation with 128-unit hidden layers.

<br/>

### Why TD3 and Not Something Else?

[TD3](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018) is purpose-built for continuous control. It extends DDPG with three innovations that address instabilities in training high-dimensional continuous policies:

| Innovation | What It Solves | How |
|:--|:--|:--|
| **Twin Critics** | Q-value overestimation compounds across many action dims | Takes $\min(Q_1, Q_2)$ — pessimistic estimate breaks the positive feedback loop |
| **Delayed Policy Updates** | Actor chases noisy critic gradients | Actor updates every 2 critic updates, giving the value function time to stabilize |
| **Target Policy Smoothing** | Brittle value peaks in action space | Adds clipped Gaussian noise to target actions: $\tilde{a} = \pi_{\theta'}(s') + \text{clip}(\epsilon, -c, c)$ |

<br/>

---

<br/>

## Environment Design

<br/>

### State Space

$$s_t = \Big[\; r_t^{\text{avail}}, \; d_1, \; d_2, \; \dots, \; d_n \;\Big]$$

A single vector concatenating the **stochastically sampled available resource** at time $t$ with the **remaining demand** for each of $n$ subtasks. At production scale: $\dim(s) = 645$.

### Action Space

$$a_t \in [0, 1]^n \quad \xrightarrow{\text{softmax}} \quad \text{allocation}_t = \text{softmax}(a_t) \cdot r_t^{\text{avail}}$$

The agent outputs continuous allocation *preferences*. Softmax normalization ensures the total allocation never exceeds available resources — a hard constraint baked into the architecture, not learned.

### Reward

| Stage | Terminal Reward (all demands satisfied) | Step Penalty |
|:--|:--|:--|
| Stage 1 (6 subtasks) | +100 | -1 per timestep |
| Stage 2 (644 subtasks) | +1000 | -1 per timestep |

The step penalty creates a gradient toward speed. The terminal bonus is large enough to dominate the cumulative penalty, ensuring the agent prioritizes *completion* over *avoidance*.

<br/>

---

<br/>

## Two-Stage Validation

The system is validated at two scales to separate algorithmic correctness from scaling behavior.

<br/>

### Stage 1 — Proof of Concept · 6 Subtasks

A compact environment validates the algorithm's core learning mechanics:

| Aspect | Detail |
|:--|:--|
| Subtasks | 6 (synthetic) |
| Network | 3-layer (256 → 256 → out) |
| Episodes | 2,000 × 2,000 max steps |
| Resources | 1 pool, stochastic availability ∈ [1, 20] |
| Evaluation | Trained vs. random actor across 2 dependency configs |

**Notebook:** `Dynamic_resource_allocation_TD3.ipynb`

<br/>

### Stage 2 — Production Scale · 644 Subtasks × 4 Resources

The algorithm is then tested against production-grade scheduling data loaded from CSV:

| Aspect | Detail |
|:--|:--|
| Subtasks | 644 (from real task decomposition) |
| Network | 4-layer (256 → 256 → 128 → out) |
| Episodes | 5,000 × 5,000 max steps |
| Resources | 4 independent pools (R1–R4), real capacity data |
| Constraints | Full precedence graph from `sub_task_successor.csv` |

**Data pipeline:**

```
task_subtask.csv        →  Decompose tasks into 644 subtasks
sub_task_successor.csv  →  Build precedence constraint graph
duration.csv            →  Subtask execution durations
resources.csv           →  R1–R4 pool capacities
requirements.csv        →  644 × 4 resource consumption matrix
```

> **Note:** The CSV data files are not included in this repository. You need to provide your own scheduling data in the format above.

**Notebook:** `RL alloc.ipynb`

<br/>

---

<br/>

## Results

<br/>

### Trained TD3 vs. Random Policy

**Stage 1 (6 subtasks, 5 resource pools):**

| Dependency Config | Trained TD3 | Random Policy |
|:--|:--|:--|
| Simple (order1) | 1,334 steps | 2,840 steps |
| Complex (order2) | 1,530 steps | 4,313 steps |

**Stage 2 (644 subtasks, 4 resource pools):**

| Metric | Trained TD3 | Random Policy |
|:--|:--|:--|
| **Steps to Complete All Subtasks** | 6,192 | 10,144 |

The trained agent completes all subtasks significantly faster than the random baseline in both stages. In Stage 2, the trained policy finishes in roughly 61% of the time the random policy requires.

> **Note:** Precedence constraints are enforced by the evaluation wrapper (`Resouce_Allocator`), which prevents both trained and random policies from allocating to blocked subtasks. The trained agent's advantage comes from learning more efficient allocation patterns for unblocked subtasks.

### Training Dynamics

- Random exploration is used for the first **10,000 steps** before the policy begins training
- Average reward generally improves across training episodes, though with typical RL noise
- The trained policy consistently outperforms the random baseline across different dependency configurations and scales

<br/>

---

<br/>

## Project Structure

```
.
├── Actor.py                                # Policy network (sigmoid output → [0,1]ⁿ)
├── Critic.py                               # Twin Q-networks (independent Q₁, Q₂)
├── TD3.py                                  # Full TD3 algorithm: train, select_action, save/load
├── Replay_Buffer.py                        # Circular experience replay (1M transitions)
├── Utils.py                                # Soft/hard target updates, softmax
├── imports.py                              # Shared dependencies (torch, numpy, etc.)
│
├── Dynamic_resource_allocation_TD3.ipynb    # Stage 1: Proof of concept (6 subtasks)
├── RL alloc.ipynb                          # Stage 2: Production scale (644 subtasks)
│
├── README.md
└── LICENSE
```

> **Note:** The standalone module files (`Actor.py`, `Critic.py`) define base architectures with 128-unit hidden layers. The notebooks define their own network classes with larger architectures tailored to each stage.

<br/>

---

<br/>

## Hyperparameters

| Parameter | Value | Why |
|:--|:--|:--|
| Discount $\gamma$ | 0.99 | Long horizon — completion can be thousands of steps away |
| Soft update $\tau$ | 0.005 | Slow target tracking prevents value function oscillation |
| Policy noise $\sigma$ | 0.2 | Smooths value landscape around actions |
| Noise clip $c$ | 0.5 | Prevents extreme exploration in target computation |
| Policy update freq | Every 2 critic steps | Core TD3 mechanism for stable actor gradients |
| Learning rate | $3 \times 10^{-4}$ (Adam) | Standard for continuous control tasks |
| Replay buffer | $10^6$ transitions | Sufficient decorrelation for off-policy learning |
| Batch size | 100 | Balances gradient variance and throughput |
| Exploration | $\mathcal{N}(0, 0.1)$ for first 10K steps | Ensures broad initial state-action coverage |

<br/>

---

<br/>

## Quick Start

### Install

```bash
pip install numpy torch matplotlib pandas
```

### Train — Proof of Concept

Open and run all cells in `Dynamic_resource_allocation_TD3.ipynb`. Trains a 6-subtask allocator in minutes and evaluates trained vs. random policy across two dependency configurations.

### Train — Production Scale

```bash
# 1. Place your scheduling CSVs in the project root:
#    task_subtask.csv, sub_task_successor.csv, duration.csv,
#    resources.csv, requirements.csv
#
# 2. Run the production notebook:
jupyter notebook "RL alloc.ipynb"
#
# 3. Trained weights saved as: RL_alloc_actor, RL_alloc_critic, etc.
```

> **Note:** Training and inference are done within the notebooks, which define their own network architectures. The standalone module files (`Actor.py`, `Critic.py`, `TD3.py`) provide a base TD3 implementation with different default layer sizes.

<br/>

---

<br/>

## The Math

<details>
<summary><strong>Click to expand full mathematical formulation</strong></summary>

<br/>

**Critic update** — minimize TD error for both $Q_1$ and $Q_2$:

$$\mathcal{L}_{\text{critic}} = \frac{1}{|B|} \sum_{(s,a,r,s') \in B} \left( Q_i(s,a) - y \right)^2$$

where the Bellman target uses the **minimum** of both target critics:

$$y = r + \gamma \cdot \min_{i=1,2} Q_{\theta'_i}\!\left(s',\; \tilde{a}'\right), \qquad \tilde{a}' = \pi_{\phi'}(s') + \text{clip}(\epsilon,\, -c,\, c), \quad \epsilon \sim \mathcal{N}(0, \sigma)$$

**Actor update** — delayed, every $d$ critic steps:

$$\mathcal{L}_{\text{actor}} = -\frac{1}{|B|} \sum_{s \in B} Q_1\!\left(s,\; \pi_\phi(s)\right)$$

**Target networks** — Polyak averaging:

$$\theta' \leftarrow \tau\,\theta + (1 - \tau)\,\theta'$$

**Action post-processing** — softmax normalization against available budget:

$$\text{alloc}_t = \frac{e^{a_i}}{\sum_j e^{a_j}} \cdot r_t^{\text{avail}}$$

This ensures $\sum_i \text{alloc}_{t,i} = r_t^{\text{avail}}$ — a hard feasibility constraint enforced by architecture, not learned.

</details>

<br/>

---

<br/>

## Key Design Decisions

| Decision | Rationale |
|:--|:--|
| **Sigmoid actor output** instead of tanh | Actions are allocation fractions ∈ [0,1], not forces ∈ [-1,1] |
| **Softmax post-processing** | Guarantees total allocation ≤ available resources (hard constraint) |
| **Stochastic resource availability** | Forces the agent to generalize across varying budget conditions |
| **Sparse terminal reward + dense step penalty** | Balances completion incentive with urgency pressure |
| **4-layer networks at scale** | 644-dim action space needs more representational capacity than the 6-dim proof-of-concept |
| **CPU-only for production notebook** | Inference-time allocation is latency-insensitive; avoids GPU memory pressure for large state vectors |

<br/>

---

<br/>

## Extending This Work

Some natural directions if you want to build on this:

- **Multi-agent allocation** — Partition subtasks across multiple TD3 agents with shared critics
- **Curriculum learning** — Train on small dependency graphs, progressively increase to full 644-subtask problem
- **Constraint-aware architectures** — Encode the precedence graph as a GNN and feed structural embeddings to the actor
- **Online adaptation** — Fine-tune the pretrained policy on live scheduling data with smaller learning rates
- **Benchmark against OR baselines** — Compare against CPLEX/Gurobi LP relaxations and priority-rule heuristics

<br/>

---

<br/>

## References

1. Fujimoto, S., Hoof, H., & Meger, D. (2018). [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477). *ICML 2018.*
2. Lillicrap, T. P., et al. (2016). [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971). *ICLR 2016.*
3. Silver, D., et al. (2014). [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf). *ICML 2014.*

<br/>

---

<br/>

<div align="center">

**Built with PyTorch** · **MIT Licensed**

<br/>

<sub>If this project helped your research or engineering work, consider giving it a ⭐</sub>

</div>

<br/>

---

this readme is AI assisted generated, so check for mistakes
