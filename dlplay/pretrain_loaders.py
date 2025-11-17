import torch


def default_pretrain_loader(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if path:
        model.load_state_dict(torch.load(path))
    return model
