import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCNNPolicy(nn.Module):
    """
    Two-branch CNN over short sequences of state/features and button history.

    Inputs (from dataset.SequenceDataset):
      state   : (B, C_s, T, F_s)
      buttons : (B, C_b, T, B)

    Output:
      logits  : (B, K) for K actions
    """
    def __init__(self, num_actions: int, in_channels_state: int, in_channels_buttons: int):
        super().__init__()
        # State branch (hp/x/etc.)
        self.state_cnn = nn.Sequential(
            nn.Conv2d(in_channels_state, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 64, 1, 1)
        )
        # Button history branch
        self.buttons_cnn = nn.Sequential(
            nn.Conv2d(in_channels_buttons, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 64, 1, 1)
        )
        # Policy head
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, state: torch.Tensor, buttons: torch.Tensor) -> torch.Tensor:
        # Expect (B, C, T, F). If a single sample is passed, add batch dim.
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if buttons.dim() == 3:
            buttons = buttons.unsqueeze(0)

        s = self.state_cnn(state).flatten(1)      # (B, 64)
        b = self.buttons_cnn(buttons).flatten(1)  # (B, 64)
        x = torch.cat([s, b], dim=1)              # (B, 128)
        logits = self.head(x)                     # (B, K)
        return logits
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCNNPolicy(nn.Module):
    """
    Two-branch CNN over short sequences of state/features and button history.

    Inputs (from dataset.SequenceDataset):
      state   : (B, C_s, T, F_s)
      buttons : (B, C_b, T, B)

    Output:
      logits  : (B, K) for K actions
    """
    def __init__(self, num_actions: int, in_channels_state: int, in_channels_buttons: int):
        super().__init__()
        # State branch (hp/x/etc.)
        self.state_cnn = nn.Sequential(
            nn.Conv2d(in_channels_state, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 64, 1, 1)
        )
        # Button history branch
        self.buttons_cnn = nn.Sequential(
            nn.Conv2d(in_channels_buttons, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 64, 1, 1)
        )
        # Policy head
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, state: torch.Tensor, buttons: torch.Tensor) -> torch.Tensor:
        # Expect (B, C, T, F). If a single sample is passed, add batch dim.
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if buttons.dim() == 3:
            buttons = buttons.unsqueeze(0)

        s = self.state_cnn(state).flatten(1)      # (B, 64)
        b = self.buttons_cnn(buttons).flatten(1)  # (B, 64)
        x = torch.cat([s, b], dim=1)              # (B, 128)
        logits = self.head(x)                     # (B, K)
        return logits