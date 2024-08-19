from typing import Dict, List, Optional, Tuple
import torch

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator


class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics."""

    def __getattr__(self, key: str) -> Tensor:
        """Get a specific metric attribute."""
        # Using this you get the correct error message, an AttributeError instead of a KeyError
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Tensor) -> None:
        """Set a specific metric attribute."""
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete a specific metric attribute."""
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")
    


class MAPbyDistanceResults(BaseMetricResults):
    """Class to wrap final mAP results."""

    __slots__ = ("map", "map_per_class", "classes")



def boxes_to_center(boxes: Tensor) -> Tensor:
    """Convert xyxy boxes to center x, y locations."""
    center = (boxes[:, [0, 1]] + boxes[:, [2,3]]) / 2
    return center 


class MAPbyDistance(Metric):
    
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    detections: List[Tensor]
    detection_scores: List[Tensor]
    detection_labels: List[Tensor]
    groundtruths: List[Tensor]
    groundtruth_labels: List[Tensor]

    def __init__(
            self, 
            radius: int = 25, 
            class_metrics: bool = True) -> None:
        super().__init__()
                
        self.radius = radius
        self.class_metrics = class_metrics

        self.add_state("detections", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruths", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)



    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:

        # TODO: add input_validator

        for item in preds:
            self.detections.append(boxes_to_center(item['boxes']))
            self.detection_labels.append(item['labels'])
            self.detection_scores.append(item['scores'])

        for item in target:
            self.groundtruths.append(boxes_to_center(item['boxes']))
            self.groundtruth_labels.append(item['labels'])


    def _get_classes(self) -> List:
            """Return a list of unique classes found in ground truth and detection data."""
            if len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0:
                return torch.cat(self.detection_labels + self.groundtruth_labels).unique().tolist()
            return []
    

    def _evaluate_image_gt_no_preds(self, gt: Tensor, gt_label_mask: Tensor) -> Dict[str, Tensor]:
        """Evaluate images with a ground truth but no predictions.

        Args:
            gt (Tensor): Tensor with ground truths.
            gt_label_mask (Tensor): Indices to filter ground truths for a specific class. 

        Dict[str, Tensor]: Dict with tensor to indicate matching detections, sorted scores,
                            and false negatives.
        """
        gt = [gt[i] for i in gt_label_mask]

        return {
            "det_matches": torch.zeros(0, dtype=torch.bool, device=self.device),
            "det_scores": torch.zeros(0, dtype=torch.float, device=self.device),
            "false_negatives": torch.tensor(len(gt), dtype=torch.int, device=self.device)
        }


    def _evaluate_image_preds_no_gt(self, det: Tensor, img_id: int, det_label_mask: Tensor) -> Dict[str, Tensor]:
        """Eavluate images with a prediction but no ground truth. 

        Args:
            det (Tensor): Tensor with predictions.
            img_id (int): Detection index.
            det_label_mask (Tensor): Indices to filter detections for a specific class.

        Dict[str, Tensor]: Dict with tensor to indicate matching detections, sorted scores,
                            and false negatives.
        """
        det = [det[i] for i in det_label_mask]
        scores = self.detection_scores[img_id]
        scores_filtered = scores[det_label_mask]
        scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
        det = [det[i] for i in dtind]

        return {
            "det_matches": torch.zeros(len(det), dtype=torch.bool, device=self.device),
            "det_scores": scores_sorted.to(self.device),
            "false_negatives": torch.tensor(0., dtype=torch.float, device=self.device)
        }
    

    def _evaluate_image(self, img_id: int, class_id: int) -> Dict[str, Tensor]:
        """Perform evaluation for single class and image.

        Args:
            img_id (int): Image id, generated from supplied samples.
            class_id (int): Class id, genereated from supplied gt and detection labels

        Returns:
            Dict[str, Tensor]: Dict with tensor to indicate matching detections, sorted scores,
                                and false negatives.
        """
        gt = self.groundtruths[img_id]
        det = self.detections[img_id]
        gt_label_mask = (self.groundtruth_labels[img_id] == class_id).nonzero().squeeze(1)
        det_label_mask = (self.detection_labels[img_id] == class_id).nonzero().squeeze(1)

        # No Gt and No predictions --> ignore image
        if len(gt_label_mask) == 0 and len(det_label_mask) == 0:
            return None
        
        # Some GT but no predictions
        if len(gt_label_mask) > 0 and len(det_label_mask) == 0:
            return self._evaluate_image_gt_no_preds(gt, gt_label_mask)

        # Some predictions but no GT
        if len(gt_label_mask) == 0 and len(det_label_mask) >= 0:
            return self._evaluate_image_preds_no_gt(det, img_id, det_label_mask)

        gt = [gt[i] for i in gt_label_mask]
        det = [det[i] for i in det_label_mask]
        if len(gt) == 0 and len(det) == 0:
            return None
        if isinstance(det, dict):
            det = [det]
        if isinstance(gt, dict):
            gt = [gt]

        scores = self.detection_scores[img_id]
        scores_filtered = scores[det_label_mask]
        scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
        det = [det[i] for i in dtind]

        det_matches = torch.zeros(len(det), dtype=torch.bool, device=self.device)
        for d_idx, d in enumerate(det):
            dist = torch.cdist(d.unsqueeze(0), torch.stack(gt, dim=0), p=2)
            min_dist, min_idx = torch.min(dist, dim=1)
            if min_dist < self.radius:
                det_matches[d_idx] = 1
                gt.pop(min_idx)
                if len(gt) == 0:
                    break

        false_negatives = len(gt_label_mask) - det_matches.sum()

        return {
            "det_matches": det_matches.to(self.device),
            "det_scores": scores_sorted.to(self.device),
            "false_negatives": false_negatives.to(self.device)
        }
        

        
    def _calculate(self, class_ids: List[int]) -> Tensor:
            """Calculate the average precision for all supplied classes to calculate mAP.

            Args:
                class_ids:
                    List of label class Ids.
            """
            img_ids = range(len(self.groundtruths))

            eval_imgs = [
                self._evaluate_image(img_id, class_id) 
                for class_id in class_ids
                for img_id in img_ids
                ]
            
            nb_classes = len(class_ids)
            nb_imgs = len(img_ids)
            ap = -torch.ones((nb_classes))

            for idx_class, _ in enumerate(class_ids):
                ap = MAPbyDistance._calculate_recall_precision_scores(
                    ap,
                    idx_class=idx_class,
                    nb_imgs=nb_imgs,
                    eval_imgs=eval_imgs
                )

            return ap
    
    
            

    @staticmethod
    def _calculate_recall_precision_scores(
            ap: Tensor, 
            idx_class: int,
            nb_imgs: int,
            eval_imgs: list
            ) -> Tuple[Tensor, Tensor, Tensor]:
        idx_cls_pointer = idx_class * nb_imgs
        # load all image evals for currnent class_id
        img_eval_cls = [eval_imgs[idx_cls_pointer + i] for i in range(nb_imgs)]
        img_eval_cls = [e for e in img_eval_cls if e is not None]
        if not img_eval_cls:
            return ap
        
        det_scores = torch.cat([e["det_scores"] for e in img_eval_cls])
        dtype = torch.uint8 if det_scores.is_cuda and det_scores.dtype is torch.bool else det_scores.dtype
        # Explicitly cast to uint8 to avoid error for bool inputs on CUDA to argsort
        inds = torch.argsort(det_scores.to(dtype), descending=True)
        det_scores_sorted = det_scores[inds]

        det_matches = torch.cat([e["det_matches"] for e in img_eval_cls])[inds]
        false_negatives = torch.tensor([e["false_negatives"] for e in img_eval_cls]).sum()

        cumulative_tp = torch.cumsum(det_matches, dim=0, dtype=torch.float)
        cumulative_tpfp = torch.arange(1, len(det_matches)+1, dtype=torch.float, device=det_matches.device)
        total_positive = det_matches.sum() + false_negatives

        eps = torch.tensor(torch.finfo(torch.float64).eps, device=det_matches.device)
        pr = cumulative_tp / (cumulative_tpfp + eps)
        rc = cumulative_tp / (total_positive + eps)

        ap[idx_class], _, _, _ = MAPbyDistance._calculate_all_points_average_precision(pr, rc)

        return ap




    @staticmethod
    def _calculate_all_points_average_precision(
            precision: Tensor,
            recall: Tensor
    ) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
        """All-point interploated average precision.

        Args:
            precision (Tensor): Tensor of all-point precision values.
            recall (Tensor): Tensor of all-point recall values.

        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]: 
                Average precision,
                interpolated recall,
                interpolated precision,
                interpolated points

        """
        recall = torch.cat((torch.tensor([0.], device=recall.device), recall, torch.tensor([1.], device=recall.device)))
        precision = torch.cat((torch.tensor([0.], device=precision.device), precision, torch.tensor([0.], device=precision.device)))

        for i in range(len(precision) - 1, 0, -1):
            precision[i-1] = torch.max(precision[i-1], precision[i])

        ii = []
        for i in range(len(recall) - 1):
            if recall[i+1] != recall[i]:
                ii.append(i+1)
        
        ap = torch.tensor(0.0, device=recall.device)
        for i in ii:
            ap += torch.sum((recall[i] - recall[i-1]) * precision[i])

        return ap.item(), recall[0:len(precision)-1].tolist(), \
                    precision[0:len(precision)-1].tolist(), ii
    



    def _summarize_results(
            self,
            precisions: Tensor, 
            recalls: Tensor
    ) -> MAPbyDistanceResults:
        pass





    def _summarize(
            self,
            results: Dict[str, Tensor]
    ) -> Tensor:
        pass 
        





    def compute(self) -> Dict[str, Tensor]:
        """Computes the metric.

        Returns:
            Dict[str, Tensor]: _description_
        """
        classes = self._get_classes()
        ap_metrics = self._calculate(classes)

        map_metrics = MAPbyDistanceResults()
        map_metrics.map = torch.tensor([-1.0]) if len(ap_metrics[ap_metrics > -1]) == 0 else torch.mean(ap_metrics[ap_metrics > -1])

        map_per_class_values: Tensor = torch.tensor([-1.0])
        if self.class_metrics:
            map_per_class_values = ap_metrics
        map_metrics.map_per_class = map_per_class_values
        map_metrics.classes = torch.tensor(classes, dtype=torch.int)
        return map_metrics






if __name__ == "__main__":
    preds_dict = {
        'boxes': torch.tensor([[0,0,5,5], [10,10,15,15], [20,20,25,25], [30,30,35,35]]),
        'scores': torch.tensor([0.8, 0.7, 0.5, 0.9]),
        'labels': torch.tensor([1, 1, 1, 1])
    }

    target_dict = {
        'boxes': torch.tensor([[0,0,5,5], [10,10,15,15], [20,20,25,25], [30,30,35,35]]),
        'labels': torch.tensor([1, 1, 1, 1])
    }      

    radius = 1
    ap = MAPbyDistance(radius=radius, class_metrics=False)

    ap.update([preds_dict], [target_dict])

    map_metrics = ap.compute()

    print(map_metrics)
