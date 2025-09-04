"""
This file contains a custom implementation of the metrics used in the evaluation of the TrackRad2025 Challenge.
The metrics are implemented as subclasses of a replica of the IterationMetric class from MONAI.
This allows us to not rely on the MONAI library for the evaluation of the algorithms,
resulting in a much more lightweight and self-contained solution.

The metrics implemented are:
- DiceMetric: computes the Dice coefficient between the predicted and ground truth segmentations.
- SurfaceDistance95Metric: computes the 95th percentile of the Hausdorff distance between the predicted and ground truth segmentations.
- SurfaceDistanceAvgMetric: computes the average surface distance between the predicted and ground truth segmentations.

The monai_metrics_test.ipybn notebook contains tests for the implemented metrics,
comparing the results with the MONAI library to ensure correctness/equality.
"""
import numpy as np
from typing import Any, Sequence
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt, binary_erosion

TensorOrList = Sequence[np.ndarray] | np.ndarray

class IterationMetric():
    """
    Base class for metrics computation at the iteration level, that is, on a min-batch of samples
    usually using the model outcome of one iteration.

    `__call__` is designed to handle `y_pred` and `y` (optional) in np.ndarrays or a list/tuple of np.ndarrays.

    Subclasses typically implement the `_compute_tensor` function for the actual tensor computation logic.
    """

    def __call__(
        self, y_pred: TensorOrList, y: TensorOrList | None = None, **kwargs: Any
    ) -> np.ndarray | Sequence[np.ndarray | Sequence[np.ndarray]]:
        """
        Execute basic computation for model prediction `y_pred` and ground truth `y` (optional).
        It supports inputs of a list of "channel-first" Tensor and a "batch-first" Tensor.

        Args:
            y_pred: the raw model prediction data at one iteration, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            y: the ground truth to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            kwargs: additional parameters for specific metric computation logic (e.g. ``spacing`` for SurfaceDistanceMetric, etc.).

        Returns:
            The computed metric values at the iteration level.
            The output shape could be a `batch-first` tensor or a list of `batch-first` tensors.
            When it's a list of tensors, each item in the list can represent a specific type of metric.

        """
        # handling a list of channel-first data
        if isinstance(y_pred, (list, tuple)) or isinstance(y, (list, tuple)):
            return self._compute_list(y_pred, y, **kwargs)
        # handling a single batch-first data
        if isinstance(y_pred, np.ndarray):
            return self._compute_tensor(y_pred, y, **kwargs)
        raise ValueError("y_pred or y must be a list/tuple of `channel-first` Tensors or a `batch-first` Tensor.")

    def _compute_list(
        self, y_pred: TensorOrList, y: TensorOrList | None = None, **kwargs: Any
    ) -> np.ndarray | list[np.ndarray | Sequence[np.ndarray]]:
        """
        Execute the metric computation for `y_pred` and `y` in a list of "channel-first" tensors.

        The return value is a "batch-first" tensor, or a list of "batch-first" tensors.
        When it's a list of tensors, each item in the list can represent a specific type of metric values.
        """
        if y is not None:
            ret = [
                self._compute_tensor(p.detach().unsqueeze(0), y_.detach().unsqueeze(0), **kwargs)
                for p, y_ in zip(y_pred, y)
            ]
        else:
            ret = [self._compute_tensor(p_.detach().unsqueeze(0), None, **kwargs) for p_ in y_pred]

        # concat the list of results (e.g. a batch of evaluation scores)
        if isinstance(ret[0], np.ndarray):
            return np.concatenate(ret, dim=0)  # type: ignore[arg-type]
        # the result is a list of sequence of tensors (e.g. a batch of multi-class results)
        if isinstance(ret[0], (list, tuple)) and all(isinstance(i, np.ndarray) for i in ret[0]):
            return [np.concatenate(batch_i, axis=0) for batch_i in zip(*ret)]
        return ret

    def _compute_tensor(self, y_pred: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> TensorOrList:
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class DiceMetric(IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        B, C, H, W = y_pred.shape
        dice = np.empty((B, C), dtype=np.float32)
        
        for b, c in np.ndindex(B, C):
            seg_gt = y_true[b, c]
            seg_pred = y_pred[b, c]
            volume_sum = seg_gt.sum() + seg_pred.sum()
            if volume_sum == 0:
                dice[b, c] = np.nan
            volume_intersect = np.logical_and(seg_gt, seg_pred).sum()
            dice[b, c] = 2*volume_intersect / volume_sum
        
        return dice
       
class HausdorffDistanceMetric(IterationMetric):
    def __init__(self, percentile=95, spacing=[1.0, 1.0], directed=False) -> None:
        super().__init__()
        self.percentile = percentile
        self.spacing = spacing
        self.directed = directed

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        assert y_true.shape == y_pred.shape, f"y_pred and y_true should have same shapes, got {y_pred.shape} and {y_true.shape}."
        
        B, C, H, W = y_pred.shape
        
        hd = np.empty((B, C), dtype=np.float32)

        for b, c in np.ndindex(B, C):
            seg_pred, seg_gt = y_pred[b, c], y_true[b, c]
            
            seg_union = np.logical_or(seg_pred,seg_gt)
            if not seg_union.any(): # not seg_pred.any() or not seg_gt.any():
                hd[b, c] = np.nan
                continue
            
            
            # compute the bounding box of the union of the two segmentations
            a = np.argwhere(seg_union)
            bb_union = np.array((np.min(a[:, 0]), np.min(a[:, 1]), np.max(a[:, 0]), np.max(a[:, 1])))
            # add a margin to the bounding box
            margin = 1
            bb_union += np.array((-margin, -margin, margin, margin))
            # clip to the image size
            bb_union = np.clip(bb_union, 0, seg_pred.shape[-2])

            # crop the bounding box to the minimum size for efficiency
            s = np.s_[bb_union[0]:bb_union[2], bb_union[1]:bb_union[3]]
            seg_pred, seg_gt = seg_pred[s], seg_gt[s]
            
            # compute the edges of the segmentations
            edges_pred = np.logical_xor(binary_erosion(seg_pred), seg_pred)
            edges_gt = np.logical_xor(binary_erosion(seg_gt), seg_gt)

            # if no edges are present, the distance is infinite
            if not edges_gt.any() or not edges_pred.any():
                hd[b, c] = np.inf
                continue

            # compute the distance transform with scipy distance_transform_edt
            distances = scipy_distance_transform_edt(
                input=(~edges_gt), 
                sampling=self.spacing
            )
            # compute the houseforff distance
            distances = distances.astype(np.float32)[edges_pred]
            hd[b, c] = np.quantile(distances, self.percentile / 100)

            # if directed, compute the distance from the other direction and take the maximum
            if not self.directed:
                distances2 = scipy_distance_transform_edt(
                    input=(~edges_pred), 
                    sampling=self.spacing
                )
                distances2 = distances2.astype(np.float32)[edges_gt]
                hd[b, c] = max(hd[b, c], np.quantile(distances2, self.percentile / 100))

        return hd

class SurfaceDistanceMetric(IterationMetric):
    def __init__(self, spacing = [1.0, 1.0]) -> None:
        super().__init__()
        self.spacing = spacing

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        assert y_true.shape == y_pred.shape, f"y_pred and y_true should have same shapes, got {y_pred.shape} and {y_true.shape}."
        

        B, C, H, W = y_pred.shape
        asd = np.empty((B, C), dtype=np.float32)

        for b, c in np.ndindex(B, C):
            seg_pred, seg_gt = y_pred[b, c], y_true[b, c]
            
            seg_union = np.logical_or(seg_pred,seg_gt)
            if not seg_union.any(): # not seg_pred.any() or not seg_gt.any():
                asd[b, c] = np.nan
                continue
            
            # compute the bounding box of the union of the two segmentations
            a = np.argwhere(seg_union)
            bb_union = np.array((np.min(a[:, 0]), np.min(a[:, 1]), np.max(a[:, 0]), np.max(a[:, 1])))
            # add a margin to the bounding box
            margin = 1
            bb_union += np.array((-margin, -margin, margin, margin))
            # clip to the image size
            bb_union = np.clip(bb_union, 0, seg_pred.shape[-2])

            # crop the bounding box to the minimum size for efficiency
            s = np.s_[bb_union[0]:bb_union[2], bb_union[1]:bb_union[3]]
            seg_pred, seg_gt = seg_pred[s], seg_gt[s]
            
            # compute the edges of the segmentations
            edges_pred = np.logical_xor(binary_erosion(seg_pred), seg_pred)
            edges_gt = np.logical_xor(binary_erosion(seg_gt), seg_gt)
            
            # if no edges are present, the distance is infinite
            if not edges_gt.any() or not edges_pred.any():
                asd[b, c] = np.inf
                continue

            # compute the distance transform with scipy distance_transform_edt
            distances = scipy_distance_transform_edt(
                input=(~edges_gt), 
                sampling=self.spacing
            )
            
            # compute the average surface distance
            distances = distances.astype(np.float32)[edges_pred]
            asd[b, c] = np.mean(distances)

        return asd
    