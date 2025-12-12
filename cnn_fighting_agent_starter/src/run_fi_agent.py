import argparse
from collections import deque

import numpy as np
import torch
from py4j.java_gateway import JavaGateway, GatewayParameters

from model import MultiCNNPolicy


def _compat_remap_state_dict(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    remapped = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("state_cnn.0.0.", "state_cnn.0.")
        nk = nk.replace("state_cnn.1.0.", "state_cnn.2.")
        nk = nk.replace("button_cnn.0.0.", "buttons_cnn.0.")
        nk = nk.replace("button_cnn.1.0.", "buttons_cnn.2.")
        nk = nk.replace("button_cnn.", "buttons_cnn.")
        nk = nk.replace("fc.0.", "head.0.")
        nk = nk.replace("fc.3.", "head.2.")
        remapped[nk] = v
    return remapped


def _load_checkpoint(path: str):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and len(sd) and all(str(k).startswith("model.") for k in sd.keys()):
        sd = {k[len("model.") :]: v for k, v in sd.items()}
    sd = _compat_remap_state_dict(sd)

    if "head.2.bias" in sd:
        num_actions = int(sd["head.2.bias"].shape[0])
    elif "head.2.weight" in sd:
        num_actions = int(sd["head.2.weight"].shape[0])
    else:
        raise KeyError("Checkpoint missing head.2.* keys; cannot infer num_actions")

    return sd, num_actions


def _connect(host: str, port: int):
    gw = JavaGateway(gateway_parameters=GatewayParameters(address=host, port=port, auto_convert=True))
    ep = gw.entry_point

    reset_fn = getattr(ep, "reset", None)
    step_fn = getattr(ep, "step", None)
    if reset_fn is None or step_fn is None:
        ctrl = ep.getEntryPoint()
        reset_fn = ctrl.reset
        step_fn = ctrl.step

    return ep, reset_fn, step_fn


def _state_to_features(state: dict) -> np.ndarray:
    my_hp = float(state.get("my_hp", 0.0))
    opp_hp = float(state.get("opp_hp", 0.0))
    my_x = float(state.get("my_x", 0.0))
    opp_x = float(state.get("opp_x", 0.0))
    dist = abs(opp_x - my_x)

    # normalize ~[0,1]
    return np.array(
        [my_hp / 400.0, opp_hp / 400.0, my_x / 960.0, opp_x / 960.0, dist / 960.0],
        dtype=np.float32,
    )


def _one_hot(idx: int, dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v


def _load_action_names(num_actions: int):
    # use config.ACTIONS if available; otherwise fallback
    try:
        from config import ACTIONS as CONFIG_ACTIONS
    except Exception:
        CONFIG_ACTIONS = []

    if len(CONFIG_ACTIONS) >= num_actions:
        return list(CONFIG_ACTIONS[:num_actions])

    fallback = [
        "idle",
        "left",
        "right",
        "jump",
        "crouch",
        "light_punch",
        "heavy_punch",
        "light_kick",
        "heavy_kick",
        "block_high",
    ]
    if len(fallback) >= num_actions:
        return fallback[:num_actions]

    return [str(i) for i in range(num_actions)]


def _action_to_env_token(action_name: str, my_x: float, opp_x: float) -> str:
    # Forward/back chosen by relative position
    opp_right = opp_x > my_x

    try:
        from fi_action_map import ACTION_TO_ENV
    except Exception:
        ACTION_TO_ENV = {}

    if action_name == "idle":
        return ACTION_TO_ENV.get("idle", "NEUTRAL")

    if action_name == "right":
        return "FORWARD" if opp_right else "BACK"
    if action_name == "left":
        return "BACK" if opp_right else "FORWARD"

    if action_name == "block_high":
        return "BACK"
    if action_name == "block_low":
        return ACTION_TO_ENV.get("block_low", "DOWN_BACK")

    return ACTION_TO_ENV.get(action_name, action_name)


def _pick_action_id(logits, action_names, my_x, opp_x, last_actions, engage_dist):
    pred = int(logits.argmax(dim=1).item())
    dist = abs(float(opp_x - my_x))

    # If far: force move toward opponent (prevents idle/block forever)
    if dist > engage_dist:
        return 2 if opp_x > my_x else 1  # right/left in the common mapping

    # If stuck: force a poke
    if len(last_actions) >= 12 and all(a == pred for a in last_actions) and action_names[pred] in ("block_high", "idle"):
        if "light_punch" in action_names:
            return action_names.index("light_punch")

    return pred


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd, num_actions = _load_checkpoint(args.weights)
    action_names = _load_action_names(num_actions)

    model = MultiCNNPolicy(num_actions=num_actions, in_channels_state=1, in_channels_buttons=1).to(device).eval()
    model.load_state_dict(sd, strict=False)

    ep, reset_fn, step_fn = _connect(args.host, args.port)
    print("Connected?", bool(ep))

    T = 12
    button_dim = 12

    wins = 0

    for epi in range(args.episodes):
        state = dict(reset_fn(args.role))
        if args.verbose:
            print(f"\n[episode {epi}] reset ->", state)

        feat_hist = deque([np.zeros((5,), np.float32) for _ in range(T)], maxlen=T)
        btn_hist = deque([np.zeros((button_dim,), np.float32) for _ in range(T)], maxlen=T)
        last_actions = deque([], maxlen=24)

        feat_hist.append(_state_to_features(state))

        done = bool(state.get("done", False))
        steps = 0

        while not done and steps < args.max_steps:
            steps += 1

            s_np = np.stack(list(feat_hist), axis=0)  # (T,5)
            b_np = np.stack(list(btn_hist), axis=0)   # (T,12)

            s = torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,5)
            b = torch.from_numpy(b_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,12)

            with torch.no_grad():
                logits = model(s, b)

            my_x = float(state.get("my_x", 0.0))
            opp_x = float(state.get("opp_x", 0.0))

            aid = _pick_action_id(logits, action_names, my_x, opp_x, last_actions, args.engage_dist)
            last_actions.append(aid)

            a_name = action_names[aid] if 0 <= aid < len(action_names) else str(aid)
            cmd = _action_to_env_token(a_name, my_x, opp_x)

            state = dict(step_fn(cmd))
            done = bool(state.get("done", False))

            feat_hist.append(_state_to_features(state))
            btn_hist.append(_one_hot(aid, button_dim))

            if args.verbose and (steps <= 10 or steps % args.print_every == 0 or done):
                print(
                    f"step {steps:4d} | aid={aid:2d} ({a_name}) cmd={cmd:>10s} | "
                    f"my_hp={state.get('my_hp')} opp_hp={state.get('opp_hp')} my_x={state.get('my_x')} opp_x={state.get('opp_x')} done={done}"
                )

        my_hp = float(state.get("my_hp", 0.0))
        opp_hp = float(state.get("opp_hp", 0.0))
        if my_hp > opp_hp:
            wins += 1

    print(f"\nEpisodes: {args.episodes}")
    print(f"Win rate: {wins/args.episodes:.3f} ({wins}/{args.episodes})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4242)
    p.add_argument("--role", default="P2")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--weights", required=True)
    p.add_argument("--engage_dist", type=float, default=170.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print_every", type=int, default=25)
    run(p.parse_args())