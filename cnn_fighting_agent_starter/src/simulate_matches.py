import argparse, random, numpy as np, pandas as pd
from fi_action_map import ACTIONS

# Build id mapping from the single source of truth
K = len(ACTIONS)
AID = {name: i for i, name in enumerate(ACTIONS)}

# Allow either long or short action names transparently
ALIASES = {
    "LP": "light_punch",  "HP": "heavy_punch",
    "LK": "light_kick",   "HK": "heavy_kick",
    # handle the reverse direction too, in case ACTIONS uses short names
    "light_punch": "LP",  "heavy_punch": "HP",
    "light_kick": "LK",   "heavy_kick": "HK",
    # common typo/old key
}

def _id(name: str) -> int:
    if name in AID:
        return AID[name]
    alt = ALIASES.get(name)
    if alt and alt in AID:
        return AID[alt]
    raise KeyError(f"Action '{name}' not found. ACTIONS={ACTIONS}")

ARENA_W = 960

# Shorthand IDs (consistent with ACTIONS order)
IDLE   = _id("idle")
LEFT   = _id("left")
RIGHT  = _id("right")
JUMP   = _id("jump")
CROUCH = _id("crouch")
LP     = _id("light_punch")
HP     = _id("heavy_punch")
LK     = _id("light_kick")
HK     = _id("heavy_kick")
BH     = _id("block_high")
BL     = _id("block_low")

def choose_action(x1, x2, d, opp_attack, hp1, hp2):
    r = random.random()
    if d <= 90:
        if opp_attack and r < 0.70:  # block more when under attack
            return BH if r < 0.85 else BL
        if (not opp_attack) and r < 0.25:
            return BH if r < 0.625 else BL
        if r < 0.55:
            return HP if r < 0.75 else HK
        if r < 0.85:
            return LP if r < 0.93 else LK
        return IDLE if r < 0.96 else JUMP
    if d <= 180:
        if r < 0.45:
            return RIGHT if x1 < x2 else LEFT
        if r < 0.65:
            return LP if r < 0.55 else LK
        if r < 0.90:
            return HP if r < 0.70 else HK
        return JUMP
    if r < 0.70:
        return RIGHT if x1 < x2 else LEFT
    if r < 0.90:
        return JUMP
    return IDLE

def step_pos(x, action):
    sp = 8
    if action == LEFT:
        x = max(0, x - sp)
    elif action == RIGHT:
        x = min(ARENA_W, x + sp)
    return x

def damage_from(action, blocked):
    if action in (LP, LK): dmg = 3
    elif action in (HP, HK): dmg = 7
    else:                   dmg = 0
    if blocked:
        dmg = max(0, dmg - 5)
    return dmg

def simulate_match(frames=300):
    x1, x2 = 200.0, 760.0
    hp1, hp2 = 100.0, 100.0
    rows = []
    for f in range(frames):
        d = abs(x2 - x1)
        r = random.random()
        if d > 200:
            opp = LEFT if x2 > x1 else RIGHT
        elif d > 80:
            opp = HP if r < 0.30 else (LEFT if (r < 0.65 and x2 > x1) else RIGHT)
        else:
            opp = HP if r < 0.60 else LP
        opp_attacking = opp in (LP, LK, HP, HK)

        act = choose_action(x1, x2, d, opp_attacking, hp1, hp2)

        x1 = step_pos(x1, act)
        x2 = step_pos(x2, opp)

        d = abs(x2 - x1)
        block1 = act in (BH, BL)
        block2 = opp in (BH, BL)
        if d <= 90:
            hp2 -= damage_from(act, block2)
        if d <= 90:
            hp1 -= damage_from(opp, block1)
        hp1 = max(0, hp1); hp2 = max(0, hp2)

        rows.append({'frame': f, 'hp_p1': hp1, 'hp_p2': hp2, 'x_p1': x1, 'x_p2': x2,
                     'distance': abs(x2-x1), 'action_id': act})
        if hp1 <= 0 or hp2 <= 0:
            break
    return rows

def main(matches, frames, val_frac, train_out, val_out, seed):
    random.seed(seed); np.random.seed(seed)
    all_rows = []
    for m in range(matches):
        rows = simulate_match(frames)
        for r in rows:
            r['frame'] = r['frame'] + m*10000
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    print('Label counts:', df['action_id'].value_counts().sort_index().to_dict())
    cut = int(len(df)*(1.0 - val_frac))
    df.iloc[:cut].to_csv(train_out, index=False)
    df.iloc[cut:].to_csv(val_out, index=False)
    print('Wrote', train_out, 'rows:', cut)
    print('Wrote', val_out,   'rows:', len(df)-cut)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, default=150)
    ap.add_argument('--frames',  type=int, default=300)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--train_out', default='data/train.csv')
    ap.add_argument('--val_out',   default='data/val.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    main(args.matches, args.frames, args.val_frac, args.train_out, args.val_out, args.seed)
import argparse, random, numpy as np, pandas as pd
from fi_action_map import ACTIONS

# Build id mapping from the single source of truth
K = len(ACTIONS)
AID = {name: i for i, name in enumerate(ACTIONS)}

# Allow short aliases -> long names
ALIASES = {
    "LP": "light_punch", "HP": "heavy_punch",
    "LK": "light_kick",  "HK": "heavy_kick",
}

def _id(name: str) -> int:
    if name in AID:
        return AID[name]
    alt = ALIASES.get(name)
    if alt and alt in AID:
        return AID[alt]
    raise KeyError(f"Action '{name}' not found. ACTIONS={ACTIONS}")

ARENA_W = 960

# Shorthand IDs (consistent with ACTIONS order)
IDLE   = _id("idle")
LEFT   = _id("left")
RIGHT  = _id("right")
JUMP   = _id("jump")
CROUCH = _id("crouch")
LP     = _id("light_punch")
HP     = _id("heavy_punch")
LK     = _id("light_kick")
HK     = _id("heavy_kick")
BH     = _id("block_high")
BL     = _id("block_low")

def choose_action(x1, x2, d, opp_attack, hp1, hp2):
    r = random.random()
    if d <= 90:
        if opp_attack and r < 0.70:  # block more when under attack
            return BH if r < 0.85 else BL
        if (not opp_attack) and r < 0.25:
            return BH if r < 0.625 else BL
        if r < 0.55:
            return HP if r < 0.75 else HK
        if r < 0.85:
            return LP if r < 0.93 else LK
        return IDLE if r < 0.96 else JUMP
    if d <= 180:
        if r < 0.45:
            return RIGHT if x1 < x2 else LEFT
        if r < 0.65:
            return LP if r < 0.55 else LK
        if r < 0.90:
            return HP if r < 0.70 else HK
        return JUMP
    if r < 0.70:
        return RIGHT if x1 < x2 else LEFT
    if r < 0.90:
        return JUMP
    return IDLE

def step_pos(x, action):
    sp = 8
    if action == LEFT:
        x = max(0, x - sp)
    elif action == RIGHT:
        x = min(ARENA_W, x + sp)
    return x

def damage_from(action, blocked):
    if action in (LP, LK): dmg = 3
    elif action in (HP, HK): dmg = 7
    else:                   dmg = 0
    if blocked:
        dmg = max(0, dmg - 5)
    return dmg

def simulate_match(frames=300):
    x1, x2 = 200.0, 760.0
    hp1, hp2 = 100.0, 100.0
    rows = []
    for f in range(frames):
        d = abs(x2 - x1)
        r = random.random()
        if d > 200:
            opp = LEFT if x2 > x1 else RIGHT
        elif d > 80:
            opp = HP if r < 0.30 else (LEFT if (r < 0.65 and x2 > x1) else RIGHT)
        else:
            opp = HP if r < 0.60 else LP
        opp_attacking = opp in (LP, LK, HP, HK)

        act = choose_action(x1, x2, d, opp_attacking, hp1, hp2)

        x1 = step_pos(x1, act)
        x2 = step_pos(x2, opp)

        d = abs(x2 - x1)
        block1 = act in (BH, BL)
        block2 = opp in (BH, BL)
        if d <= 90:
            hp2 -= damage_from(act, block2)
        if d <= 90:
            hp1 -= damage_from(opp, block1)
        hp1 = max(0, hp1); hp2 = max(0, hp2)

        rows.append({'frame': f, 'hp_p1': hp1, 'hp_p2': hp2, 'x_p1': x1, 'x_p2': x2,
                     'distance': abs(x2-x1), 'action_id': act})
        if hp1 <= 0 or hp2 <= 0:
            break
    return rows

def main(matches, frames, val_frac, train_out, val_out, seed):
    random.seed(seed); np.random.seed(seed)
    all_rows = []
    for m in range(matches):
        rows = simulate_match(frames)
        for r in rows:
            r['frame'] = r['frame'] + m*10000
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    print('Label counts:', df['action_id'].value_counts().sort_index().to_dict())
    cut = int(len(df)*(1.0 - val_frac))
    df.iloc[:cut].to_csv(train_out, index=False)
    df.iloc[cut:].to_csv(val_out, index=False)
    print('Wrote', train_out, 'rows:', cut)
    print('Wrote', val_out,   'rows:', len(df)-cut)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', type=int, default=150)
    ap.add_argument('--frames',  type=int, default=300)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--train_out', default='data/train.csv')
    ap.add_argument('--val_out',   default='data/val.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    main(args.matches, args.frames, args.val_frac, args.train_out, args.val_out, args.seed)