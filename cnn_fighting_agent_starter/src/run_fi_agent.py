import torch
import argparse
from py4j.java_gateway import JavaGateway, GatewayParameters

# --- Compatibility helpers for older checkpoints --------------------------------
def _compat_remap_state_dict(sd: dict) -> dict:
    """
    Remap older checkpoint keys to current MultiCNNPolicy names.

    Old -> New examples:
      state_cnn.0.0.*  -> state_cnn.0.*
      state_cnn.1.0.*  -> state_cnn.2.*
      button_cnn.0.0.* -> buttons_cnn.0.*
      button_cnn.1.0.* -> buttons_cnn.2.*
      button_cnn.*     -> buttons_cnn.*      # singular -> plural safety
      fc.0.*           -> head.0.*
      fc.3.*           -> head.2.*
    """
    if not isinstance(sd, dict):
        return sd
    remapped = {}
    for k, v in sd.items():
        nk = k
        # state branch (old had extra ".0" sub-blocks)
        nk = nk.replace('state_cnn.0.0.', 'state_cnn.0.')
        nk = nk.replace('state_cnn.1.0.', 'state_cnn.2.')
        # buttons branch: "button_cnn" -> "buttons_cnn" and remove extra ".0"
        nk = nk.replace('button_cnn.0.0.', 'buttons_cnn.0.')
        nk = nk.replace('button_cnn.1.0.', 'buttons_cnn.2.')
        nk = nk.replace('button_cnn.', 'buttons_cnn.')
        # head: "fc" -> "head"
        nk = nk.replace('fc.0.', 'head.0.')
        nk = nk.replace('fc.3.', 'head.2.')
        remapped[nk] = v
    return remapped

def _load_weights_compat(model, path, device):
    """
    Load a checkpoint, tolerating older key names (strict=False) and printing leftovers.
    """
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = _compat_remap_state_dict(sd)
    res = model.load_state_dict(sd, strict=False)
    try:
        missing = getattr(res, 'missing_keys', [])
        unexpected = getattr(res, 'unexpected_keys', [])
    except Exception:
        missing, unexpected = [], []
    if missing or unexpected:
        print(f"[weights] loaded with missing={missing} unexpected={unexpected}")
    return model


# --- Py4J stub connection for FightingICE --------------------------------------
def _connect_and_loop(args):
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        address=args.host, port=args.port, auto_convert=True
    ))
    ep = gw.entry_point
    # prefer direct methods; if not present, use the Controller
    reset_fn = getattr(ep, "reset", None)
    step_fn = getattr(ep, "step", None)
    if reset_fn is None or step_fn is None:
        ctrl = ep.getEntryPoint()
        reset_fn = ctrl.reset
        step_fn = ctrl.step

    print("Connected?", bool(ep))
    state = reset_fn(args.role)
    print("reset ->", dict(state))
    for i in range(args.episodes):
        state = step_fn("FORWARD")
        print(f"step {i} ->", dict(state))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4242)
    p.add_argument("--role", default="P2")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--weights", default=None)  # accepted but unused in this quick stub
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    _connect_and_loop(args)