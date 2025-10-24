# src/fi_action_map.py
ACTIONS = [
    "idle", "left", "right", "jump", "crouch",
    "LP", "HP", "LK", "HK", "block_high", "block_low",
]

# Map our action names to the FightingICE control you have.
# If your bridge wants key names, put keys; if it wants enums, put those.
ACTION_TO_ENV = {
    "idle"      : "IDLE",
    "left"      : "LEFT",         # e.g. walk/back depending on side; adjust in runner
    "right"     : "RIGHT",        # e.g. walk/forward depending on side; adjust in runner
    "jump"      : "UP",
    "crouch"    : "DOWN",
    "LP"        : "A",
    "HP"        : "B",
    "LK"        : "C",
    "HK"        : "D",            # or whatever your scheme uses
    "block_high": "BLOCK_HIGH",   # sometimes this is just holding BACK
    "block_low" : "BLOCK_LOW"    # often DOWN+BACK
     # maybe "QCF+A" in some wrappers; fill in your command
}
# src/fi_action_map.py
# NOTE: The order of ACTIONS **must** match the training order (config / dataset).
#       This order corresponds to the starter we trained:
#       0:idle, 1:left, 2:right, 3:jump, 4:crouch,
#       5:light_punch, 6:heavy_punch, 7:light_kick, 8:heavy_kick,
#       9:fireball, 10:block_high, 11:block_low

ACTIONS = [
    "idle", "left", "right", "jump", "crouch",
    "light_punch", "heavy_punch", "light_kick", "heavy_kick",
     "block_high", "block_low",
]

# Map our action names -> FightingICE control tokens.
# Replace the VALUES with the real keys/enums your FI bridge expects.
# (run_fi_agent.py will flip BACK/FORWARD for P2 and will convert block_low
#  to DOWN_BACK or DOWN_FORWARD depending on facing.)
ACTION_TO_ENV = {
    "idle"        : "NEUTRAL",
    "left"        : "BACK",        # for P1; runner flips for P2
    "right"       : "FORWARD",     # for P1; runner flips for P2
    "jump"        : "UP",
    "crouch"      : "DOWN",

    "light_punch" : "A",
    "heavy_punch" : "B",
    "light_kick"  : "C",
    "heavy_kick"  : "D",

       # or whatever your bridge uses for fireball

    # Blocks (runner may rewrite block_low to down_forward depending on side)
    "block_high"  : "BACK",
    "block_low"   : "DOWN_BACK",

    # Optional aliases (in case other parts still refer to short names)
    "LP"          : "A",
    "HP"          : "B",
    "LK"          : "C",
    "HK"          : "D",

    # Aliases used by the runner when adjusting for facing
    "down_back"   : "DOWN_BACK",
    "down_forward": "DOWN_FORWARD",
}