# PRD: Gevva0 Autonomous Arcade Pac-Man (Production Showcase)

## 1. Document Overview & Executive Summary

This Product Requirements Document (PRD) defines the architecture, telemetry pipeline, tactical heuristics, and UI specifications for the real-time **Gevva0 Autonomous Pac-Man Showcase**.

The showcase demonstrates **Project Gevva0** (Gemma 4 26B MoE running on local CUDA via `llama.cpp`) playing classic arcade Pac-Man at human-to-superhuman skill levels without game freezes, frame drops, or simulated approximations.

### Core Target

Clear Level 1 and loop through successive maze levels entirely under live Gevva0 fast-path logit control, maintaining a locked **60 FPS** UI render loop, an effective decision latency of **0 ms at turn execution**, and a model forward-pass latency of **$\le$ 70 ms**.

---

## 2. Problem Statement & Root Cause of Prior Failures

Previous iterations failed to clear Level 1 and suffered 150–300 ms perceived latencies due to four structural bottlenecks:

1. **The "Every-Grid-Square" Query Anti-Pattern**: Querying the model 70 ms prior to every square tile while Pac-Man travels at 4.2 tiles/sec (238 ms/tile) meant any inference spike over 70 ms resulted in collision or sudden stops.


2. **Cold Context Prefill Overhead**: Sending full prompt templates (200–450 tokens) on each call forced continuous KV regeneration, incurring a 40–80 ms prefill penalty on Gemma 26B MoE before logit readout.


3. **GIL Contention via Software Rasterization**: Running Pygame software rendering (`SDL_RENDER_DRIVER = "software"`) alongside CPU raycasting and the inference thread starved Python’s Global Interpreter Lock (GIL), inflating round-trip times by 80–150 ms.


4. **Uncalibrated CoT Triggering on Hazards**: Lowering CoT thresholds during ghost pursuit triggered bounded autoregressive text generation (80–150 ms) exactly when sub-30 ms reflexes were required.



---

## 3. Product Goals & Key Performance Indicators (KPIs)

| Metric | Target | Measurement Method |
| --- | --- | --- |
| **Level Progression** | Clear Level 1 consistently; survive $\ge 3$ consecutive boards | Game loop state machine |
| **UI Render Pacing** | Locked 60 FPS (16.6 ms/frame) | Pygame display clock

 |
| **Effective Execution Latency** | **0 ms** (pre-buffered turn ready at junction arrival) | Game tick delta to execution

 |
| **Gevva0 Raw Inference** | p50 $\le 65\text{ ms}$, p95 $\le 110\text{ ms}$ | High-resolution CUDA timers

 |
| **Energizer Exploitation Rate** | $\ge 2$ ghosts eaten per energizer consumed | World event log

 |
| **Illegal / Wall Move Rate** | **0.0%** (deterministic pre-pruning) | Corridor validator

 |

---

## 4. System Architecture: Dual-Layer Decoupled Engine

The architecture completely separates continuous high-speed spatial physics from high-level tactical routing.

```
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 1: LOCAL MOTOR REFLEX ENGINE (60 Hz)               │
│                     (Process A: Isolated Game Core)                    │
│                                                                        │
│ - Continuous straightaway propulsion (4.2 tiles/sec)                   │
│ - Automatic 90° cornering through hallway bends                        │
│ - Immediate 180° retreat if dangerous ghost < 1.0 tile head-on         │
│ - Pellet/Energizer consumption and score accounting                   │
│ - Waypoint Predictive Trigger (Dispatches next junction decision)      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ IPC Unix Domain Socket / Shared Memory
                                   ▼ (Serialized 28-token Delta)
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 2: GEVVA0 TACTICAL DECISION GATEWAY                │
│                   (Process B: GPU Inference Gateway)                   │
│                                                                        │
│ - Static Rule & Maze Topology KV-Cache Prefix (Preloaded once)         │
│ - Legal Move Pruning (Walls & Fatal Dead-Ends stripped out)           │
│ - Fast-Path Direct Logit Scoring over Legal Candidates [A, B, C]       │
│ - Platt Temperature Calibration (T = 1.155)                           │
│ - Output: Buffered target direction placed into `pacman.next_dir`      │
└────────────────────────────────────────────────────────────────────────┘

```

### 4.1. Process Isolation (Zero GIL Starvation)

The Pygame rendering loop and the `GemmaDecisionEngine` run as two independent OS processes communicating via non-blocking IPC (Unix Domain Socket or POSIX Shared Memory):

* **Game Process**: Runs at 60 FPS uninhibited.
* **Inference Process**: Retains exclusive CUDA GPU execution and handles KV cache manipulation without stalling the display clock.



---

## 5. Predictive Move Dispatching & KV-Cache Management

### 5.1. Predictive Junction Pre-Dispatch (Eliminating Traversal Delays)

In a standard Pac-Man maze, decision points only exist at **junctions** (tiles with 3 or 4 legal exits). Straight corridors and single corners require no tactical branching:

1. **Waypoint Calculation**: When Pac-Man exits Junction $J_k$ into a corridor leading toward Junction $J_{k+1}$, the destination coordinate $(x_{k+1}, y_{k+1})$ is calculated immediately.
2. **Immediate Background Dispatch**: Long before Pac-Man arrives (hallways take 1.2 to 3.0 seconds to traverse), the gateway dispatches the tactical state query for Junction $J_{k+1}$.
3. **Pre-Buffered Latch**: The model returns its choice within 60–80 ms. The result is written to `pacman.next_dir`.


4. **Seamless Transit**: When Pac-Man physically intersects $(x_{k+1}, y_{k+1})$, the turn executes instantly with **0 ms effective wait time**.



### 5.2. Stateful KV-Cache Prefix Management

To reduce inference latency from 110 ms down to 25–45 ms, Gevva0’s `kv_cache_seq_cp` primitive is utilized:

* **Base Sequence 0 (Static System Prompt)**: Ingests the game directives, scoring priorities, and board geometry **once** at engine launch. This context (~350 tokens) remains pinned in VRAM across the entire play session.


* **Dynamic Suffix Forking**: On each junction dispatch, a micro-sequence clones Sequence 0 and appends only the ultra-compact dynamic telemetry delta (< 45 tokens). Logits are read directly, and the temporary sequence is purged via `kv_cache_seq_rm`.



---

## 6. Board State Representation & Spatial Prompt Minimization

To ensure sub-60 ms evaluations, large verbose matrices and multi-line descriptions are replaced with an information-dense, token-minimized telemetry block:

### 6.1. Toroidal Wrap-Around Distance Metric

All distance calculations must adhere to wrap-around corridor physics:

$$\Delta x = \min(\vert{}x_1 - x_2\vert{}, \text{COLS} - \vert{}x_1 - x_2\vert{})$$

$$\Delta y = \vert{}y_1 - y_2\vert{}$$

$$\text{Dist}_{\text{toroidal}} = \sqrt{\Delta x^2 + \Delta y^2}$$

Corridors opening into warp tunnels are labeled explicitly as `WARP_TUNNEL` with exit destinations mapped directly to the opposite flank.

### 6.2. Strategic Energizer Logic (Human-Level Play)

A human player does not grab an Energizer immediately upon seeing it. The feature extractor must enforce the classic arcade heuristic:

* **Bait State**: An Energizer is rated high priority **only** if $\ge 2$ active chase ghosts are within 5 tiles of Pac-Man.
* **Conservation State**: If ghosts are scattered or $> 10$ tiles away, the Energizer corridor is deprioritized to preserve the power pellet for future escapes.

### 6.3. Dynamic Telemetry Delta (Full Dispatch Payload)

```text
[STATE] P:(10,16) D:WEST | B:(12,08) K:(09,11) I:(14,12) C:(05,18) | E_ACTIVE:NO (0.0s) | NEAR_E:(01,01) D:8 BAIT:READY
[JUNCTION (06,16)]
Candidates:
A: NORTH | Pellets: 12 | Ghosts: CLEAR | Egress: TURNS_EAST
B: WEST  | Pellets: 0  | Ghosts: BLINKEY_CLOSING (dist 3.2) | Egress: DEAD_END
C: SOUTH | Pellets: 18 | Ghosts: CLEAR | Egress: WARP_TUNNEL
Choice:

```

### 6.4. Strict Option Pruning

The candidate list presented to Gevva0 **never** contains illegal moves:

* Directions hitting walls are eliminated before token generation.


* Hallways occupied by non-frightened ghosts $\le 1.5$ tiles away are stripped out as fatal hazards.


* The candidate set is restricted to 2 or 3 legal actions (`[A]`, `[B]`, `[C]`), minimizing softmax entropy and guaranteeing single-pass fast-path execution.



---

## 7. UI & Mission Control Panel Specifications

The UI consists of a dual-viewport arcade screen rendered at 1100x750 resolution.

```
┌─────────────────────────────────┬──────────────────────────────────────────┐
│                                 │ GEVVA0 TELEMETRY : GEMMA 4 26B           │
│         1UP   HIGH SCORE   DOTS │ ┌──────────────────────────────────────┐ │
│       03420     333330      142 │ │ HAZARD: THREAT PURSUIT (Blinky 3.2t) │ │
│                                 │ └──────────────────────────────────────┘ │
│      ┌───────────────────┐      │ PELLET RADAR: 98/240 [=====>        ] 41%│
│      │ . . . . . . . . . │      │                                          │
│      │ . WWW . WWW . WWW │      │ JUNCTION DISPATCH CANDIDATES:            │
│      │ O ... P ... K ..  │      │ ┌──────────────────────────────────────┐ │
│      │ . WWW . WWW . WWW │      │ │ [A] ▲ NORTH  [42.4%] █████▍          │ │
│      │                   │      │ │ 12 Pellets | Egress: TURNS_EAST      │ │
│      │     [ ARCADE ]    │      │ ├──────────────────────────────────────┤ │
│      │     [  MAZE  ]    │      │ │ [B] ◀ WEST   [ 1.2%]                 │ │
│      │                   │      │ │ Lethal Hazard: Blinky closing        │ │
│      │                   │      │ ├──────────────────────────────────────┤ │
│      │                   │      │ │ [C] ▼ SOUTH  [56.4%] ███████▌ [WIN]  │ │
│      │                   │      │ │ 18 Pellets | Warp Tunnel Path        │ │
│      │                   │      │ └──────────────────────────────────────┘ │
│      │                   │      │                                          │
│      │ O ............. O │      │ CONFIDENCE: 89.2% (PLATT T=1.155)        │
│      └───────────────────┘      │ LATENCY: 22.4 ms | FAST-PATH VERDICT     │
│      LIVES: ● ●   DEVOURED: 4   │ FPS: 60 (CPU) | DISPATCHES: 48           │
└─────────────────────────────────┴──────────────────────────────────────────┘

```

### 7.1. Decision Illumination & UI States

* **Pre-Query State**: When a junction is dispatched in the background, the candidate box displays a cyan pulsing highlight.
* **Verdict Flash**: The instant the decision arrives, the chosen card illuminates in neon-yellow (normal route) or neon-green (energizer/predation route). The unselected cards dim to 20% opacity.


* **Real-Time Probability Bars**: Displays calibrated posterior distributions calculated via Platt temperature scaling ($T = 1.1550$):



$$P(y = c) = \frac{\exp(z_c / T)}{\sum_k \exp(z_k / T)}$$

* **Hardware Latency Monitor**: Real-time display showing true engine forward-pass latency in milliseconds, mode (`FAST_PATH` or `FORCED_MOVE`), and total dispatches.



---

## 8. Implementation Roadmap

### Phase 1: Engine & IPC Decoupling

* Extract game logic into an isolated, standalone process.
* Implement non-blocking Unix Domain Socket IPC for dispatch queries and verdict buffers.
* Eliminate Pygame GIL lockup by binding display updates strictly to the rendering loop.



### Phase 2: KV Cache Sequence Pinned Session

* Configure `llama.cpp` context to retain static game rules and board topology in Sequence 0.


* Wire `kv_cache_seq_cp` to fork Sequence 0 on dispatches, evaluating only the dynamic delta.


* Benchmark forward-pass latency to ensure p50 $\le 65\text{ ms}$ on local GPU.



### Phase 3: Tactical Raycaster & Warp Math

* Implement toroidal distance routines for warp tunnels.
* Implement human-style Energizer baiting state checks (require pursuit count $\ge 2$ within 5 tiles before targeting power pellets).
* Implement the Junction Predictive Trigger to fire when entering a hallway instead of 70 ms before an intersection.

### Phase 4: UI Telemetry Polish & Level Validation

* Connect real-time logit bars and illumination cards to IPC responses.


* Run continuous automated sessions to validate reliable clears of Level 1 through Level 3.