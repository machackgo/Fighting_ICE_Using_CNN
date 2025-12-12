import argparse
import random
import numpy as np
import torch

from config import ACTION_TO_ID as AID, KEY_TO_ACTION as KTA


def main(weights, episodes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # toy model: random actions
    class DummyModel:
        def __init__(self):
            self.num_actions = len(AID)

        def __call__(self, s, b):
            batch_size = s.shape[0]
            logits = torch.randn(batch_size, self.num_actions)
            return logits

    model = DummyModel()

    wins = 0

    for ep_i in range(episodes):
        my_hp = 400
        opp_hp = 400
        my_x = 200
        opp_x = 760
        done = False
        steps = 0

        while not done and steps < 600:
            steps += 1

            # dummy state features (not used)
            s = torch.zeros((1, 1, 12, 5)).to(device)
            b = torch.zeros((1, 1, 12, 12)).to(device)

            logits = model(s, b)
            action_id = int(logits.argmax(dim=1).item())
            action = list(AID.keys())[action_id]

            # simple toy dynamics
            if action == "right":
                my_x = min(my_x + 7, 960)
            elif action == "left":
                my_x = max(my_x - 7, 0)

            # simple attack logic
            dist = abs(my_x - opp_x)
            if dist < 80 and action in ("light_punch", "heavy_punch", "light_kick", "heavy_kick"):
                opp_hp -= 10

            # simple opponent
            opp_action = random.choice(["idle", "left", "right", "light_punch", "heavy_punch"])
            if opp_action == "right":
                opp_x = min(opp_x + 7, 960)
            elif opp_action == "left":
                opp_x = max(opp_x - 7, 0)

            if dist < 80 and opp_action in ("light_punch", "heavy_punch", "light_kick", "heavy_kick"):
                my_hp -= 10

            if my_hp <= 0 or opp_hp <= 0 or steps >= 600:
                done = True

        if my_hp > opp_hp:
            wins += 1

        print(f"Episode {ep_i}: steps={steps} my_hp={my_hp} opp_hp={opp_hp} win={my_hp > opp_hp}")

    print(f"\nEpisodes: {episodes}")
    print(f"Win rate: {wins/episodes:.3f} ({wins}/{episodes})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    main(args.weights, args.episodes)
