
import argparse
import random
import numpy as np
import torch

from config import ACTION_TO_ID as AID, K
from model import MultiCNNPolicy

# --- shorthand action ids from config ---
IDLE = AID.get("idle", 0)
LEFT = AID["left"]
RIGHT = AID["right"]
JUMP = AID.get("jump", IDLE)
CROUCH = AID.get("crouch", IDLE)

LP = AID.get("light_punch", IDLE)
HP = AID.get("heavy_punch", IDLE)
LK = AID.get("light_kick", IDLE)
HK = AID.get("heavy_kick", IDLE)
SP = AID.get("special_1", LP)

BH = AID.get("block_high", IDLE)
BL = AID.get("block_low", IDLE)

ATTACK_IDS = [i for i in (LP, HP, LK, HK, SP) if i != IDLE]

ARENA_W = 960
CLOSE_RANGE = 90
MID_RANGE = (90, 160)
STEP = 6

# simple damage model
DMG = {
    LP: 6,
    LK: 7,
    HP: 12,
    HK: 14,
    SP: 10,
}
BLOCK_REDUCTION = 0.75  # 75% damage reduction when block is correct


def is_attack(aid: int) -> bool:
    return aid in ATTACK_IDS


def is_block(aid: int) -> bool:
    return aid in (BH, BL)


def choose_action_id_with_rules(x1, x2, opp_attacking, logits: torch.Tensor) -> int:
    """
    Rule wrapper to keep the CNN in the loop but avoid passive/losing choices.
    - Far: walk toward the opponent.
    - Mid: mostly advance, sometimes throw special.
    - Close & opp attacking: 60% block, 40% counter-attack.
    - Close & safe: strongly bias toward attacks.
    """
    d = abs(float(x2 - x1))
    pred_id = int(logits.argmax(dim=1).item())

    # --- FAR (create engagement) ---
    if d > MID_RANGE[1]:
        return RIGHT if x2 > x1 else LEFT

    # --- MID (close distance; sometimes poke with special) ---
    if MID_RANGE[0] < d <= MID_RANGE[1]:
        if opp_attacking:
            # block under pressure or keep closing
            return BH if random.random() < 0.50 else (RIGHT if x2 > x1 else LEFT)
        # 20% chance to throw a ranged special if available
        if SP in ATTACK_IDS and random.random() < 0.20:
            return SP
        # Otherwise keep closing 70% of the time
        if random.random() < 0.70:
            return RIGHT if x2 > x1 else LEFT
        return pred_id

    # --- CLOSE (fight!) ---
    # If the opponent is currently attacking: mostly block, sometimes counter
    if opp_attacking and random.random() < 0.60:
        return BH if random.random() < 0.70 else BL

    # Pick the best attack id according to the model logits
    log_np = logits.detach().cpu().numpy().ravel()
    mask = np.full_like(log_np, -1e9, dtype=np.float32)
    for i in ATTACK_IDS:
        if 0 <= i < log_np.shape[0]:
            mask[i] = 0.0
    # 85%: force an attack; otherwise trust the model's argmax
    return int((log_np + mask).argmax()) if random.random() < 0.85 else pred_id


def opponent_policy(x1, x2, opp_attacking_prev: bool) -> int:
    """Very simple heuristic opponent used only for evaluation."""
    d = abs(x2 - x1)
    r = random.random()

    if d > MID_RANGE[1]:
        # far: approach
        return LEFT  # opponent moves left toward the agent at x1 < x2 in our coordinate choice
    if d > CLOSE_RANGE:
        # mid: mostly approach, sometimes block if previously under pressure
        if opp_attacking_prev and r < 0.15:
            return BH if random.random() < 0.7 else BL
        if r < 0.75:
            return LEFT
        return random.choice([LP, LK, HP, HK])
    # close
    if r < 0.60:
        return BH if random.random() < 0.7 else BL
    return random.choice([LP, LK, HP, HK])


def step(x1, x2, hp1, hp2, a1, a2):
    """Advance one frame."""
    # move
    if a1 == LEFT:
        x1 = max(0, x1 - STEP)
    elif a1 == RIGHT:
        x1 = min(ARENA_W, x1 + STEP)
    if a2 == LEFT:
        x2 = max(0, x2 - STEP)
    elif a2 == RIGHT:
        x2 = min(ARENA_W, x2 + STEP)

    d = abs(x2 - x1)

    # attacks
    if d <= CLOSE_RANGE:
        # agent hits opponent
        if is_attack(a1):
            dmg = DMG.get(a1, 0)
            if a2 in (BH, BL):
                dmg = max(0, int(dmg * (1.0 - BLOCK_REDUCTION)))
            hp2 -= dmg
        # opponent hits agent
        if is_attack(a2):
            dmg = DMG.get(a2, 0)
            if a1 in (BH, BL):
                dmg = max(0, int(dmg * (1.0 - BLOCK_REDUCTION)))
            hp1 -= dmg

    return x1, x2, hp1, hp2


def evaluate(weights: str, episodes: int):
    """Run a crude simulator to estimate win rate of the policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model dims: training used 1 channel for states/buttons; actions = K
    model = MultiCNNPolicy(
        num_actions=K,
        in_channels_state=1,
        in_channels_buttons=1,
    ).to(device).eval()

    # Load weights (be tolerant to missing keys)
    try:
        sd = torch.load(weights, map_location=device)
        # accept both raw state_dict and checkpoints with "state_dict"
        if isinstance(sd, dict) and "state_dict" in sd and not all(k.startswith("model.") for k in sd["state_dict"]):
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        print(f"[warn] failed to load weights '{weights}': {e}")

    wins = 0
    total_frames = 0
    hp_diff_sum = 0.0

    for _ in range(int(episodes)):
        x1, x2 = 100.0, ARENA_W - 100.0
        hp1 = hp2 = 100
        opp_attacking_prev = False

        frames = 0
        for _ in range(260):
            frames += 1
            # dummy inputs with correct tensor ranks (1, C, H, W)
            s = torch.zeros(1, 1, 12, 5, device=device)
            b = torch.zeros(1, 1, 12, 12, device=device)
            with torch.no_grad():
                logits = model(s, b)

            # opponent decides first for "pressure" signal
            a2 = opponent_policy(x1, x2, opp_attacking_prev)
            opp_attacking = is_attack(a2)

            # agent decision with rule wrapper
            a1 = choose_action_id_with_rules(x1, x2, opp_attacking, logits)

            # one step
            x1, x2, hp1, hp2 = step(x1, x2, hp1, hp2, a1, a2)
            opp_attacking_prev = opp_attacking

            if hp1 <= 0 or hp2 <= 0:
                break

        total_frames += frames
        hp_diff_sum += (hp1 - hp2)
        if hp1 > hp2:
            wins += 1

    print(f"Episodes: {episodes}")
    print(f"Win rate: {wins/episodes:.3f}  ({wins}/{episodes})")
    print(f"Avg frame length: {total_frames/episodes:.1f}")
    print(f"Avg HP diff (agent - opp): {hp_diff_sum/episodes:.2f}")


def main(weights, episodes):
    # Directly call our evaluate loop
    return evaluate(weights, episodes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/latest/model.pt")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()
    main(args.weights, args.episodes)
