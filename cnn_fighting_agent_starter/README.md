# Multi‑CNN Fighting Game Agent (Starter Kit)

You will build an **AI player** for a fighting game environment (e.g., FightingICE).
The agent reads short **sequences of game state + button inputs** (turned into image‑like tensors),
runs them through a **multi‑branch CNN**, and outputs the **next action** (policy).

> Scope: You are **not** building a new game. You are building a **controller/agent** that plugs into an existing engine.

## Repository layout
```
cnn_fighting_agent/
├── data/                       # Put your recorded matches (.npz) here
├── docs/
│   └── design.md               # Design notes (start here)
├── src/
│   ├── dataset.py              # Loads sequences -> tensors
│   ├── model.py                # Multi-branch CNN
│   ├── train.py                # Behavior cloning training loop
│   ├── eval.py                 # Offline evaluation
│   ├── generate_dummy_data.py  # Creates a toy dataset to test the pipeline
│   └── fightingice_runner.py   # Placeholder for live inference loop
├── outputs/                    # Trained weights, logs
├── requirements.txt
└── README.md
```

## Quick start (offline, with dummy data)
1. (Recommended) Use Python **3.10 or 3.11**. Newer (3.13) builds of PyTorch may not be available yet.
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create a small synthetic dataset and train an end‑to‑end sanity check:
   ```bash
   python src/generate_dummy_data.py --out data/samples.npz
   python src/train.py --data data/samples.npz --epochs 5 --batch_size 64
   python src/eval.py --data data/samples.npz --weights outputs/latest/model.pt
   ```

## Real data shape (behavior cloning)
We expect an `.npz` file with numpy arrays:
- `states`:  (N, C_state, T, F_state)    e.g., channels for [hp, x, y, vx, vy, distance]
- `buttons`: (N, C_buttons, T, B)        e.g., one‑hot button history
- `labels`:  (N,)                         integer action id (0..K‑1)

The network predicts the **next action** after each T‑length window.

## Going live (later)
- Implement `src/fightingice_runner.py` to read the environment's observation and maintain sliding windows.
- Call the trained PyTorch model to get an action each step and send it to the engine.

## Deliverables to your team
- Data pipeline + loader
- Multi‑CNN model and training scripts
- Trained weights + offline metrics
- Short demo video/GIF of agent acting in the environment
- 1‑2 page write‑up summarizing your design and results


## To run this file and getting the accuracy and win rate of this modle run this prompts in terminal 

   # Quick: just run the sim with your latest model

      cd ~/Documents/cnn_fighting_agent_starter
      source .venv/bin/activate
      python -u src/sim_eval_agent.py --weights outputs/latest/model.pt --episodes 50


   # Full pipeline (regen data → convert → train → eval → sim)

      cd ~/Documents/cnn_fighting_agent_starter
      source .venv/bin/activate

      # 1) Generate synthetic match logs
      python src/simulate_matches.py --matches 300 --frames 260 --val_frac 0.15 --train_out data/train.csv --val_out data/val.csv

      # 2) Convert CSV → NPZ
      python src/prepare_from_csv.py --csv data/train.csv --out data/train.npz
      python src/prepare_from_csv.py --csv data/val.csv   --out data/val.npz

      # 3) (Optional) sanity-check label distribution
      python -c "import numpy as np, collections; y=np.load('data/train.npz')['labels']; print('class counts:', dict(sorted(collections.Counter(y.tolist()).items())))"

      # 4) Train
      python src/train.py --data data/train.npz --epochs 10 --batch_size 256

      # 5) Eval (offline accuracy)
      python src/eval.py  --data data/val.npz --weights outputs/latest/model.pt

      # 6) Simulate matches with the trained agent
      python -u src/sim_eval_agent.py --weights outputs/latest/model.pt --episodes 50

   # Save a checkpoint of a good model (optional)
      mkdir -p outputs/checkpoints
      cp outputs/latest/model.pt outputs/checkpoints/winrate_baseline.pt

 ## To run the updated version 
   cd ~/Documents/cnn_fighting_agent_starter
   source .venv/bin/activate   

   cd ~/Documents/cnn_fighting_agent_starter
   source .venv/bin/activate
   python src/simulate_matches.py \
  --matches 300 --frames 260 --val_frac 0.15 \
  --train_out data/train.csv --val_out data/val.csv  

  cd ~/Documents/cnn_fighting_agent_starter
   python src/prepare_from_csv.py --csv data/train.csv --out data/train.npz
   python src/prepare_from_csv.py --csv data/val.csv   --out data/val.npz

   cd ~/Documents/cnn_fighting_agent_starter
   python src/train.py --data data/train.npz --epochs 10 --batch_size 256

   cd ~/Documents/cnn_fighting_agent_starter
   python src/eval.py --data data/val.npz --weights outputs/latest/model.pt

   cd ~/Documents/cnn_fighting_agent_starter
   python -u src/sim_eval_agent.py --weights outputs/latest/model.pt --episodes 50


   ## 

cd ~/Documents/cnn_fighting_agent_starter
source .venv/bin/activate

cd ~/Documents/cnn_fighting_agent_starter
source .venv/bin/activate
python - <<'PY'
from py4j.java_gateway import JavaGateway, GatewayParameters
gw = JavaGateway(gateway_parameters=GatewayParameters(address='127.0.0.1', port=4242, auto_convert=True))
ep = gw.entry_point
print("Connected?", bool(ep))
print("reset ->", dict(ep.reset("P2")))
print("step  ->", dict(ep.step("FORWARD")))
PY

cd ~/Documents/cnn_fighting_agent_starter
source .venv/bin/activate
python src/simulate_matches.py --matches 300 --frames 260 --val_frac 0.15 \
  --train_out data/train.csv --val_out data/val.csv

python src/prepare_from_csv.py --csv data/train.csv --out data/train.npz
python src/prepare_from_csv.py --csv data/val.csv   --out data/val.npz

python -c "import numpy as np, collections; y=np.load('data/train.npz')['labels']; print('class counts:', dict(sorted(collections.Counter(y.tolist()).items())))"

python src/train.py --data data/train.npz --epochs 10 --batch_size 256

python src/eval.py --data data/val.npz --weights outputs/latest/model.pt

python -u src/sim_eval_agent.py --weights outputs/latest/model.pt --episodes 50

mkdir -p outputs/checkpoints
cp outputs/latest/model.pt outputs/checkpoints/winrate_baseline.pt

cd ~/Documents/cnn_fighting_agent_starter
source .venv/bin/activate
PYTHONPATH=src python -u src/run_fi_agent.py \
  --weights outputs/latest/model.pt \
  --role P2 --episodes 5 \
  --host 127.0.0.1 --port 4242 \
  --verbose

cd ~/Documents/cnn_fighting_agent_starter
source .venv/bin/activate
PYTHONPATH=src python - <<'PY'
from fi_action_map import ACTIONS
print("len(ACTIONS) =", len(ACTIONS))
print(ACTIONS)
PY








