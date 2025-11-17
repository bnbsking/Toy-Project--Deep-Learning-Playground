from collections import Counter
import copy
from typing import Dict, List, Optional, Union

import numpy as np
import sklearn.metrics as skm

from ..output_collectors import BaseOutputCollector


class BaseMetricsPipeline:
    def __init__(
            self,
            num_classes: Optional[int] = None,
            labels: Optional[List] = None,
            predictions: Optional[List] = None,
            func_dicts: Optional[List[Dict]] = None,
            **kwargs,
        ):
        self.num_classes = num_classes
        self.labels = labels
        self.predictions = predictions
        self.func_dicts = func_dicts

        self.metrics = {"loss": kwargs.get("loss", -1)}
    
    def _deserialize(self, data: Dict):
        if isinstance(data, dict):
            return {k: self._deserialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def run(self) -> Dict:
        for func_dict in self.func_dicts:
            self.metrics[func_dict["log_name"]] = \
                getattr(self, func_dict["func_name"])(**func_dict["func_args"])
        for k, v in self.metrics.items():
            self.metrics[k] = self._deserialize(v)
        return self.metrics


class RegressionMetricsPipeline(BaseMetricsPipeline):
    def __init__(
            self,
            output_collector: BaseOutputCollector,
            func_dicts: Dict,
            **kwargs
        ):
        """
        Notes:
            self.labels (np.array): shape = (n_samples, n_classes)
            self.predictions (np.array): shape = (n_samples, n_classes)
        """
        super().__init__(
            labels = output_collector.labels,
            predictions = output_collector.predictions,
            func_dicts = func_dicts,
            **kwargs
        )

    def r2_score(self) -> float:
        return skm.r2_score(self.labels.reshape(-1), self.predictions.reshape(-1))


class ClassificationMetricsPipeline(BaseMetricsPipeline):
    def __init__(
            self,
            output_collector: BaseOutputCollector,
            func_dicts: Dict,
            **kwargs
        ):
        """
        Notes:
            self.labels (np.array): shape = (n_samples, n_classes)
            self.predictions (np.array): shape = (n_samples, n_classes)
        """
        kwargs_copy = copy.deepcopy(kwargs)
        super().__init__(
            num_classes = kwargs_copy.pop("num_classes"),
            labels = output_collector.labels,
            predictions = output_collector.predictions,
            func_dicts = func_dicts,
            **kwargs_copy
        )
        self.single_label = kwargs.get("single_label", True)
        if self.single_label:
            self.labels = self.labels.reshape(-1)
        self.start_idx = self.single_label and kwargs.get("start_idx", 0)
        self.gt_class_cnts = self._get_gt_class_cnts()

    def _get_gt_class_cnts(self) -> List[int]:
        gt_class_cnts = [0] * (self.num_classes - self.start_idx)
        if self.single_label:
            for label in self.labels:
                gt_class_cnts[label - self.start_idx] += 1
        else:
            for label_list in self.labels:  # multilabel returns positive count only
                for cls_idx, is_positive in enumerate(label_list):
                    gt_class_cnts[cls_idx] += is_positive
        return gt_class_cnts

    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(self.num_classes - self.start_idx)
        ]

        for i, threshold in enumerate(np.linspace(0, 1, k)):
            for j in range(self.start_idx, self.num_classes):
                if self.single_label:
                    gt_cls = (self.labels == j).astype(np.int32)
                    pd_cls = (self.predictions[:, j] >= threshold).astype(np.int32)
                else:
                    gt_cls = (self.labels[:, j] == j).astype(np.int32)
                    pd_cls = (self.predictions[:, j, 1] >= threshold).astype(np.int32)
                precision = skm.precision_score(gt_cls, pd_cls, zero_division=0.0)
                recall = skm.recall_score(gt_cls, pd_cls, zero_division=0.0)
                pr_curves[j - self.start_idx]["precision"][i] = precision
                pr_curves[j - self.start_idx]["recall"][i] = recall

        return pr_curves
    
    def get_refine_pr_curves(self, pr_curves_key: str = "pr_curves") -> List[Dict[str, List[float]]]:
        """
        sorted by recall, and enhance precision by next element reversely
        Args:
            pr_curves_key (str): get pr_curves from self.metrics and refine it.
        Dependency:
            you must call self.get_pr_curves in advance
        """
        pr_curves = copy.deepcopy(self.metrics[pr_curves_key])
        refine_pr_curves = [{} for _ in range(len(pr_curves))]
        for i in range(len(pr_curves)):
            recall_arr = pr_curves[i]["recall"].copy()
            precision_arr = pr_curves[i]["precision"].copy()
            zip_arr = sorted(zip(recall_arr, precision_arr))
            recall_arr, precision_arr = zip(*zip_arr)
            recall_arr, precision_arr = list(recall_arr), list(precision_arr)
            for j in range(1, len(precision_arr)):
                precision_arr[-1-j] = max(precision_arr[-1-j], precision_arr[-j])
            refine_pr_curves[i]["refine_recall"] = recall_arr
            refine_pr_curves[i]["refine_precision"] = precision_arr
        return refine_pr_curves
    
    def get_ap_list(self, refine_pr_curves_key: str = "refine_pr_curves") -> List[float]:
        """
        Args:
            refine_pr_curves_key (str): get refine_pr_curves from self.metrics and compute aps
        Dependency:
            you must call self.get_refine_pr_curves in advance
        """
        refine_pr_curves = self.metrics[refine_pr_curves_key]
        k_val = len(refine_pr_curves[0]["refine_precision"])  # 101
        ap_list = []
        for i in range(len(refine_pr_curves)):
            ap = 0
            for j in range(k_val - 1):
                ap += refine_pr_curves[i]["refine_precision"][j] * \
                    (refine_pr_curves[i]["refine_recall"][j+1] - refine_pr_curves[i]["refine_recall"][j])
            ap_list.append(round(ap,3))
        return ap_list
    
    def get_map(self, ap_list_key: str) -> float:
        """
        Args:
            ap_list_key (str): get ap_list from self.metrics and compute map
        Dependency:
            you must call self.get_aps in advance
        """
        ap_list = self.metrics[ap_list_key]
        return round(sum(ap_list) / len(ap_list), 3)
    
    def get_wmap(self, ap_list_key: str) -> float:
        """
        Args:
            ap_list_key (str): get ap_list from self.metrics and compute wmap
        Dependency:
            you must call self.get_aps in advance
        """
        ap_list = self.metrics[ap_list_key]
        return round(sum(ap * cnt for ap, cnt in zip(ap_list, self.gt_class_cnts)) \
                / sum(self.gt_class_cnts), 3)
    
    def get_best_threshold(self, strategy: str = "f1", **kwargs) -> float:
        """
        get best threshold by some strategy
        Args:
            strategy (str): currently support "f1" or "precision" only
        Returns:
            best_threshold (float)
        Dependency:
            you must call self.get_pr_curves in advance
        Notes:
            For classification does not have background, the return value is meaningless
        """
        if strategy in {"f1", "precision"}:
            if strategy == "f1":
                score_func = lambda precision, recall: \
                    2 * precision * recall / (precision + recall + 1e-10)
            elif strategy == "precision":
                score_func = lambda precision, recall: \
                    precision if recall >= 0.5 else 0
            pr_curves_key = kwargs["pr_curves_key"]

            pr_curves = self.metrics[pr_curves_key]
            k_val = len(pr_curves[0]["precision"])
            thresholds = np.linspace(0, 1, k_val)  # 101
            weighted_score = [0] * len(thresholds)
            for i in range(len(pr_curves)):
                for j, (precision, recall) in enumerate(
                        zip(pr_curves[i]["precision"], pr_curves[i]["recall"])
                    ):
                    score = score_func(precision, recall)
                    weighted_score[j] += score * self.gt_class_cnts[i] / sum(self.gt_class_cnts)
            _, best_threshold = max(zip(weighted_score, thresholds))
            return best_threshold
        else:
            return 0.5
    
    def get_confusion_axis_norm(self, confusion_key: str, axis: int) -> np.ndarray:
        """
        Args:
            axis (int): either 0 (col, precision) or 1 (row, recall)
        Returns:
            confusion_axis_norm (np.ndarray[float]): shape same as input confusion
        Dependency:
            you must call self.confusion in advance
        """
        confusion_axis_norm = self.metrics[confusion_key].copy().astype(np.float32)
        axis_sum = confusion_axis_norm.sum(axis=axis)
        for i in range(len(confusion_axis_norm)):
            if axis == 0:
                confusion_axis_norm[:, i] /= (axis_sum[i] + 1e-10)
            elif axis == 1:
                confusion_axis_norm[i, :] /= (axis_sum[i] + 1e-10)
        return confusion_axis_norm

    def get_confusion(
            self,
            threshold: float = 0.5,
            threshold_key: str = ""
        ) -> np.ndarray:
        """
        Args:
            threshold (float)
            threshold_key (str): if specified, use self.metrics[threshold_key] instead of threshold
        Returns:
            confusion (np.ndarray[int]): shape=(num_classes, num_classes).
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if self.single_label and self.start_idx==0 and self.num_classes > 2:  # multiclass without background
            gt_cls = self.labels
            pd_cls = self.predictions.argmax(axis=1)
        elif self.single_label:  # binary or multiclass with background
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels
            pd_cls = np.where(
                    self.predictions[:, 0] < threshold,
                    0,
                    self.predictions[:, 1:].argmax(axis=1) + 1
                )
        else:  # multilabel
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels.reshape(-1)
            pd_cls = (self.predictions[:, :, 1] >= threshold).reshape(-1).astype(np.int32)
        
        confusion = skm.confusion_matrix(gt_cls, pd_cls, labels=list(range(self.num_classes)))
        return confusion

    def get_confusion_with_img_indices(
            self,
            threshold: float = 0.5,
            threshold_key: str = ""
        ) -> List[List[Counter[int, int]]]:
        """
        Args:
            threshold (float)
            threshold_key (str): if specified, use self.metrics[threshold_key] instead of threshold
        Returns:
            confusion_with_img_indices (List[List[Counter[int, int]]]):
                shape=(num_classes, num_classes). each grid is counter of image indices
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if self.single_label and self.start_idx==0 and self.num_classes > 2:  # multiclass without background
            gt_cls = self.labels
            pd_cls = self.predictions.argmax(axis=1)
        elif self.single_label:  # binary or multiclass with background
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels
            pd_cls = np.where(
                    self.predictions[:, 0] > threshold,
                    0,
                    self.predictions[:, 1:].argmax(axis=1) + 1
                )
        else:  # multilabel
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels.reshape(-1)
            pd_cls = (self.predictions[:, :, 1] >= threshold).reshape(-1).astype(np.int32)
        
        confusion_with_img_indices = [
            [Counter() for _ in range(self.num_classes)] for _ in range(self.num_classes)
        ]
        dataset_length = len(self.labels)
        for idx, (gtc, pdc) in enumerate(zip(gt_cls, pd_cls)):
            confusion_with_img_indices[gtc][pdc][idx % dataset_length] += 1
        return confusion_with_img_indices
