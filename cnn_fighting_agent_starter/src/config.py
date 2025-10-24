ACTIONS = [
  "idle","left","right","jump","crouch",
  "light_punch","heavy_punch","light_kick","heavy_kick",
  "block_high","block_low","special_1"
]
ACTION_TO_ID = {a:i for i,a in enumerate(ACTIONS)}
K = len(ACTIONS)

FEATURES = ["hp_p1","hp_p2","x_p1","x_p2","distance"]
T = 12
FEATURE_MAX = {"hp_p1":400, "hp_p2":400, "x_p1":960, "x_p2":960, "distance":960}