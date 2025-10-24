import argparse, os
import torch
from torch.utils.data import DataLoader
from dataset import SequenceDataset
from model import MultiCNNPolicy
from tqdm import tqdm


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and build model
    ds = SequenceDataset(args.data)
    num_actions = int(ds.labels.max()) + 1
    model = MultiCNNPolicy(
        num_actions=num_actions,
        in_channels_state=ds.states.shape[1],
        in_channels_buttons=ds.buttons.shape[1]
    ).to(device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Optimizer + stable loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total, correct, loss_sum = 0, 0, 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for (s, b), y in pbar:
            s = s.to(device)
            b = b.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(s, b)
            loss = criterion(logits, y)

            # Guard against NaN/Inf and exploding grads
            if not torch.isfinite(loss):
                print("[warn] non-finite loss detected:", float(loss.item()))
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                loss_sum += loss.item() * y.size(0)
                pbar.set_postfix(acc=f"{correct/total:.3f}", loss=f"{loss_sum/total:.2f}")

    os.makedirs("outputs/latest", exist_ok=True)
    torch.save(model.state_dict(), "outputs/latest/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .npz dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)