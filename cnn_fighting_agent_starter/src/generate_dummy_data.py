import argparse, numpy as np, os

def make_dummy(out_path, N=1200, T=12, F_state=6, B=8, C_state=3, C_buttons=1, K=12, seed=0):
    rng = np.random.default_rng(seed)
    # States: emulate simple correlated features (hp drops, distance changes)
    states = rng.random((N, C_state, T, F_state), dtype=np.float32)
    # Buttons: sparse multi-hot like presses (here: random 0/1 noise as placeholder)
    buttons = (rng.random((N, C_buttons, T, B)) > 0.9).astype(np.float32)
    # Simple rule to generate labels from features so learning is possible:
    # e.g., if last frame distance feature > 0.5 -> action 0, else action 1, plus noise.
    last_dist = states[:, 0, -1, 0]  # pick one feature as "distance"
    labels = (last_dist * K).astype(np.int64) % K
    noise = rng.integers(0, 2, size=N, dtype=np.int64)
    labels = (labels + noise) % K

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, states=states, buttons=buttons, labels=labels)
    print(f"Wrote dummy dataset to {out_path}")
    print(f"shapes: states={states.shape}, buttons={buttons.shape}, labels={labels.shape}, actions={K}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples.npz")
    ap.add_argument("--N", type=int, default=1200)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--F_state", type=int, default=6)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--K", type=int, default=12)
    args = ap.parse_args()
    make_dummy(args.out, N=args.N, T=args.T, F_state=args.F_state, B=args.B, K=args.K)