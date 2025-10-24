# src/fi_features.py
def extract_features(frame, you_are_p1=True, stage_w=960, max_hp=400):
    """
    Convert one FightingICE frame object into [hp_p1, hp_p2, x_p1, x_p2, distance] in [0,1].
    Replace the TODOs with your wrapper's actual field access.
    """
    # --- TODO: read these from your FightingICE bridge ---
    if you_are_p1:
        hp1  = frame.my_hp      # 0..max_hp
        hp2  = frame.opp_hp     # 0..max_hp
        x1   = frame.my_x       # 0..stage_w
        x2   = frame.opp_x      # 0..stage_w
    else:
        hp1  = frame.opp_hp
        hp2  = frame.my_hp
        x1   = frame.opp_x
        x2   = frame.my_x
    # ----------------------------------------------------

    # normalize
    hp1  = max(0, min(max_hp, hp1)) / float(max_hp)
    hp2  = max(0, min(max_hp, hp2)) / float(max_hp)
    x1   = max(0, min(stage_w, x1)) / float(stage_w)
    x2   = max(0, min(stage_w, x2)) / float(stage_w)
    dist = abs(x1 - x2)

    return [hp1, hp2, x1, x2, dist]