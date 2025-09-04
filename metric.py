from konfai.metric.measure import Criterion
from monai_metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, IterationMetric
import torch
import numpy as np
import scipy

class EuclideanCenterDistanceMetric(IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        B,C,H,W = y_pred.shape
        true_com_path = np.array([scipy.ndimage.center_of_mass(y_true[i,0,:,:]) for i in range(B)])
        pred_com_path = np.array([scipy.ndimage.center_of_mass(y_pred[i,0,:,:]) for i in range(B)])

        # L2 norm of the difference between the predicted and true center of mass
        return np.linalg.norm(true_com_path - pred_com_path, axis=1)
    
class TrackradMetric(Criterion):

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        y_true = output.squeeze(0).squeeze(0).numpy().transpose(2,0,1)
        y_pred = targets[0].squeeze(0).squeeze(0).numpy().transpose(2,0,1)
        T, H, W = y_true.shape

        y_pred_monai = y_pred.reshape(T,1,H,W)
        y_true_monai = y_true.reshape(T,1,H,W)

        empty_true = y_true_monai.sum(axis=(1,2,3)) == 0

        # Exclude empty true targets
        y_pred_monai = y_pred_monai[~empty_true]
        y_true_monai = y_true_monai[~empty_true]

        # Exclude empty predictions for geometric metrics
        empty_pred = y_pred_monai.sum(axis=(1,2,3)) == 0
        y_pred_monai_non_empty = y_pred_monai[~empty_pred]
        y_true_monai_non_empty = y_true_monai[~empty_pred]

        max_distance = max(H,W)
        penalty = np.full(empty_pred.sum(), max_distance)
        return self.compute(y_pred_monai_non_empty, y_true_monai_non_empty, empty_pred, penalty)

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray, empty_pred: np.ndarray, penalty: np.ndarray) -> torch.Tensor:
        pass

class Dice(TrackradMetric):

    def __init__(self):
        super().__init__()
        self.dice_metric = DiceMetric()

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray, empty_pred: np.ndarray, penalty: np.ndarray) -> torch.Tensor:
        dsc = self.dice_metric(y_pred, y_true)
        dsc = np.concatenate((dsc.flatten(), np.zeros(empty_pred.sum())))
        return dsc[1:].mean()

class HausdorffDistance(TrackradMetric):

    def __init__(self):
        super().__init__()
        self.hausdorff_distance_95_metric = HausdorffDistanceMetric(percentile=95)
    
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray, empty_pred: np.ndarray, penalty: np.ndarray) -> torch.Tensor:
        hausdorff_distance_95 = self.hausdorff_distance_95_metric(y_pred, y_true)
        hausdorff_distance_95 = np.concatenate((hausdorff_distance_95.flatten(), penalty))
        return  hausdorff_distance_95[1:].mean()

class SurfaceDistance(TrackradMetric):

    def __init__(self):
        super().__init__()
        self.surface_distance_avg_metric = SurfaceDistanceMetric()

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray, empty_pred: np.ndarray, penalty: np.ndarray) -> torch.Tensor:
        surface_distance_average = self.surface_distance_avg_metric(y_pred, y_true)
        surface_distance_average = np.concatenate((surface_distance_average.flatten(), penalty))
        return  surface_distance_average[1:].mean() 

class EuclideanCenterDistance(TrackradMetric):

    def __init__(self):
        super().__init__()
        self.center_distance_metric = EuclideanCenterDistanceMetric()

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray, empty_pred: np.ndarray, penalty: np.ndarray) -> torch.Tensor:
        center_distance = self.center_distance_metric(y_pred, y_true)
        return center_distance[1:].mean()
    