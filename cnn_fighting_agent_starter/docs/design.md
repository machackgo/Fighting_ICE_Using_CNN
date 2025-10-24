# Design Notes (High Level)

**Objective**: Learn a policy π(a|history) that maps the last T timesteps of
game state + button inputs to the next action. We use **two CNN branches**:

- **Branch A (state branch)**: 2D convolutions over a tensor shaped **(C_state, T, F_state)**
  where T is "time" (height) and F_state are features (width). This treats the
  time–feature grid like an image.

- **Branch B (buttons branch)**: 2D convolutions over **(C_buttons, T, B)** where
  B is the number of possible buttons. This helps the model learn combo patterns.

The flattened features from both branches are concatenated and passed to dense layers.
The output head is a softmax over **K actions**.

**Loss**: Cross‑entropy (behavior cloning).  
**Metrics**: Top‑1 accuracy (offline), win‑rate vs baseline (online).

**Data windowing**: Create sliding windows of length T across each match and label
each window with the next action taken by the player (expert or bot).

**Normalization**: Scale continuous features (e.g., hp, x, y, distance) to [0,1].
Buttons should be one‑hot or multi‑hot per frame.

**Ablations to try**:
- Single CNN vs multi‑CNN
- Different T (e.g., 8, 12, 16)
- With/without buttons branch