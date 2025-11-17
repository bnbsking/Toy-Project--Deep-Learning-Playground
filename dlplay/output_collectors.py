from abc import abstractmethod
import os

import numpy as np
import pandas as pd
import torch


class BaseOutputCollector:
    def __init__(self, **kwargs):
        self.labels = []
        self.predictions = [] 
        
    @abstractmethod
    def update(self, labels, predictions):
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, save_dir: str):
        raise NotImplementedError

    def reset(self):
        self.labels = []
        self.predictions = []


class DefaultOutputCollector(BaseOutputCollector):
    def update(self, label: torch.Tensor, prediction: torch.Tensor):
        """
        Collect batch inputs.
        Args:
            label (torch.Tensor): shape = (n_samples, ...)
            prediction (torch.Tensor): shape = (n_samples, ...)
        """
        self.labels.extend(label.detach().cpu().numpy())
        self.predictions.extend(prediction.detach().cpu().numpy())
    
    def postprocess(self):
        # concatenate all batch inputs
        self.labels = np.expand_dims(self.labels, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.predictions = np.expand_dims(self.predictions, axis=0)
        self.predictions = np.concatenate(self.predictions, axis=0)
    
    def save(self, save_dir: str, mode: str):
        # save as csv
        df_labels = pd.DataFrame(self.labels)
        df_predictions = pd.DataFrame(self.predictions)
        df_labels.to_csv(os.path.join(save_dir, f"{mode}_ytrue.csv"), index=False)
        df_predictions.to_csv(os.path.join(save_dir, f"{mode}_ypred.csv"), index=False)
