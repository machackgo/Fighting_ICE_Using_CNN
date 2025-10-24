import argparse, numpy as np, pandas as pd
from config import FEATURES, FEATURE_MAX, K, T

def make_windows(df):
    feats = df[FEATURES].astype(float).values  # (L, F)
    acts  = df["action_id"].astype(int).values
    # normalize
    for j, col in enumerate(FEATURES):
        mx = float(FEATURE_MAX[col])
        feats[:, j] = np.clip(feats[:, j]/mx, 0.0, 1.0)

    windows_s, windows_b, labels = [], [], []
    L, F = feats.shape
    for i in range(L - T):
        s_win = feats[i:i+T, :]           # (T, F)
        a_win = acts[i:i+T]               # (T,)
        s_img = s_win[None, :, :]         # (1, T, F)
        b_img = np.zeros((T, K), np.float32)
        for t, a in enumerate(a_win):
            if 0 <= a < K:
                b_img[t, a] = 1.0
        b_img = b_img[None, :, :]         # (1, T, K)
        y = acts[i+T]                     # next action after the window
        windows_s.append(s_img)
        windows_b.append(b_img)
        labels.append(y)

    states  = np.stack(windows_s).astype(np.float32)   # (N, 1, T, F)
    buttons = np.stack(windows_b).astype(np.float32)   # (N, 1, T, K)
    labels  = np.array(labels, dtype=np.int64)         # (N,)
    return states, buttons, labels

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    states, buttons, labels = make_windows(df)
    np.savez_compressed(args.out, states=states, buttons=buttons, labels=labels)
    print("Wrote:", args.out, "shapes:", states.shape, buttons.shape, labels.shape)
