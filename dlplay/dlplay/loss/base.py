import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, y.reshape(-1))
