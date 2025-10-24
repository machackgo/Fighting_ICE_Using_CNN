import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from dataset import SequenceDataset
from model import MultiCNNPolicy

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SequenceDataset(args.data)
    num_actions = int(ds.labels.max()) + 1
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model = MultiCNNPolicy(num_actions=num_actions,
                           in_channels_state=ds.states.shape[1],
                           in_channels_buttons=ds.buttons.shape[1]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for (s, b), y in loader:
            s, b, y = s.to(device), b.to(device), y.to(device)
            logits = model(s, b)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    print(f"Accuracy: {correct/total:.4f}  ({correct}/{total})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", required=True)
    args = p.parse_args()
    main(args)