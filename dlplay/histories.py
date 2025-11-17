from abc import abstractmethod
import json
import os
from typing import Dict, List


class BaseHistory:
    def __init__(self, save_dir: str, mode: str):
        self.history = {}
        self.save_path = os.path.join(save_dir, f"{mode}_history.json")
    
    @abstractmethod
    def update(self, data: dict):
        raise NotImplementedError

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=4)


class DefaultHistory(BaseHistory):
    def __init__(
            self,
            metrics: List[str],
            save_dir: str
        ):
        self.history = {
            metric: [] for metric in metrics
        }
        self.save_dir = save_dir

    def update(self, metrics_dict: Dict):
        for key, value in metrics_dict.items():
            self.history[key].append(value)
