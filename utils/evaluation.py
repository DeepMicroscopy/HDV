"""

Adopted from https://github.com/DeepPathology/TUPAC16_AlternativeLabels/blob/master/lib/calculate_F1.py

"""
from typing import List, Union

import numpy as np
import pandas as pd
import torch 
import json

from sklearn.neighbors import KDTree
# from scipy.spatial import KDTree
from typing import Any, Dict, Tuple
from torchvision.ops import box_iou

from evalutils.scorers import score_detection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from .map_by_distance import MAPbyDistance



def _F1_core_iou(
    annos: np.ndarray, 
    boxes: np.ndarray, 
    scores: np.ndarray, 
    det_thresh: float, 
    iou_thresh: float = 0.5) -> Tuple[float, int, int, int]:
    """Computes F1 score using IoU for matching detections with annotations, optimized with PyTorch.

    Args:
        annos (np.ndarray): array of ground truth bounding boxes in [xmin, ymin, xmax, ymax] format.
        boxes (np.ndarray): predicted bounding boxes in [xmin, ymin, xmax, ymax] format.
        scores (np.ndarray): predicted scores.
        det_thresh (float): detection score threshold.
        iou_thresh (float, optional): IoU threshold for matching. Defaults to 0.5.

    Returns:
        Tuple[float, int, int, int]: F1 score, true positives, false positives, false negatives.
    """
    # filter detections based on the detection threshold
    keep = scores > det_thresh
    boxes = boxes[keep]

    if boxes.shape[0] == 0 or annos.shape[0] == 0:
        return 0, 0, boxes.shape[0], annos.shape[0]

    # convert arrays to torch tensors
    boxes = torch.tensor(boxes, dtype=torch.float32)
    annos = torch.tensor(annos, dtype=torch.float32)

    # compute IoU between all detections and annotations
    iou_matrix = box_iou(boxes, annos)

    # match detections to annotations using the IoU threshold
    max_iou_per_detection, max_iou_indices = iou_matrix.max(dim=1)

    # track which annotations have been matched
    matched_annotations = torch.zeros(annos.shape[0], dtype=torch.bool)

    TP = 0
    for i in range(max_iou_per_detection.shape[0]):
        if max_iou_per_detection[i] >= iou_thresh:
            matched_annotation_idx = max_iou_indices[i].item()
            if not matched_annotations[matched_annotation_idx]:  # if annotation is not already matched
                TP += 1
                matched_annotations[matched_annotation_idx] = True

    # false positives are unmatched detections
    FP = (max_iou_per_detection < iou_thresh).sum().item()

    # false negatives are unmatched annotations
    FN = (~matched_annotations).sum().item()

    # Calculate F1 score
    if TP + FP + FN == 0:
        F1 = 0.0
    else:
        F1 = 2 * TP / (2 * TP + FP + FN)

    return F1, TP, FP, FN


# def _F1_core_kdtree(
#     annos: np.ndarray, 
#     boxes: np.ndarray, 
#     scores: np.ndarray, 
#     det_thresh: float, 
#     radius: int = 25) -> Tuple[float, int, int, int]:
#     """Computes F1 score for a given set of annotations and detections.

#     Args:
#         annos (np.ndarray): array of center coordinates in the format [x,y].
#         boxes (np.ndarray): predicted bounding boxes in the format [xmin,ymin,xmax,ymax].
#         scores (np.ndarray): predicted scores.
#         det_thresh (float): detection threshold.
#         radius (int, optional): radius of kd-tree query. Defaults to 25.

#     Returns:
#         Tuple[float, int, int, int]: f1, tp, fp, fn.
#     """
#     keep = scores > det_thresh
#     boxes = boxes[keep]

#     if boxes.shape[0] == 0 or annos.shape[0] == 0:
#         return 0, 0, boxes.shape[0], annos.shape[0]

#     # Calculate center coordinates for detections
#     cx_det = (boxes[:, 0] + boxes[:, 2]) / 2
#     cy_det = (boxes[:, 1] + boxes[:, 3]) / 2

#     # KDTree for detections
#     detections = np.vstack((cx_det, cy_det)).T
#     tree_det = KDTree(detections)

#     # Query each annotation to find the closest detection
#     cx_annos = annos[:, 0]
#     cy_annos = annos[:, 1]
#     annotations = np.vstack((cx_annos, cy_annos)).T

#     ind = tree_det.query(annotations, distance_upper_bound=radius)[1]

#     # Initialize TP, FP, FN
#     TP = 0
#     FN = 0
#     FP = 0

#     # Track matched detections
#     matched_detections = np.zeros(detections.shape[0], dtype=bool)

#     # Count true positives and false negatives
#     for i, idx in enumerate(ind):
#         if idx < len(detections):  # Valid index, meaning a detection was found within the radius
#             if not matched_detections[idx]:  # If detection is not already matched
#                 TP += 1
#                 matched_detections[idx] = True
#         else:
#             FN += 1  # Annotation did not find a match

#     # False positives are unmatched detections
#     FP = np.sum(~matched_detections)

#     # F1 score calculation
#     if TP + FP + FN == 0:
#         F1 = 0.0
#     else:
#         F1 = 2 * TP / (2 * TP + FP + FN)

#     return F1, TP, FP, FN


# def _F1_core_kdtree(
#     annos: np.ndarray, 
#     boxes: np.ndarray, 
#     scores: np.ndarray, 
#     det_thresh: float, 
#     radius: int = 25) -> Tuple[float, int, int, int]:
#     """Computes F1 score for a given set of annotations and detections.

#     Args:
#         annos (np.ndarray): array of center coordinates in the format [x,y].
#         boxes (np.ndarray): predicted bounding boxes in the format [xmin,ymin,xmax,ymax].
#         scores (np.ndarray): predicted scores.
#         det_thresh (float): detection threshold.
#         radius (int, optional): radius of kd-tree query. Defaults to 25.

#     Returns:
#         Tuple[float, int, int, int]: f1, tp, fp, fn.
#     """
#     keep = scores > det_thresh
#     boxes = boxes[keep]

#     cx = (boxes[:, 0] + boxes[:, 2]) / 2
#     cy = (boxes[:, 1] + boxes[:, 3]) / 2

#     isDet = np.zeros(boxes.shape[0] + annos.shape[0])
#     isDet[0:boxes.shape[0]] = 1 

#     if annos.shape[0] > 0:
#             cx = np.hstack((cx, annos[:, 0]))
#             cy = np.hstack((cy, annos[:, 1]))

#     # set up kdtree 
#     X = np.dstack((cx, cy))[0]

#     if X.shape[0] == 0:
#         return 0, 0, 0, 0

#     try:
#         tree = KDTree(X)
#     except:
#         print('Shape of X: ', X.shape)

#     ind = tree.query_radius(X, r=radius)

#     annotationWasDetected = {x: 0 for x in np.where(isDet==0)[0]}
#     DetectionMatchesAnnotation = {x: 0 for x in np.where(isDet==1)[0]}

#     # check: already used results
#     alreadyused=[]
#     for i in ind:
#         if len(i) == 0:
#             continue
#         if np.any(isDet[i]) and np.any(isDet[i]==0):
#             # at least 1 detection and 1 non-detection --> count all as hits
#             for j in range(len(i)):
#                 if not isDet[i][j]: # is annotation, that was detected
#                     if i[j] not in annotationWasDetected:
#                         print('Missing key ',j, 'in annotationWasDetected')
#                         raise ValueError('Ijks')
#                     annotationWasDetected[i[j]] = 1
#                 else:
#                     if i[j] not in DetectionMatchesAnnotation:
#                         print('Missing key ',j, 'in DetectionMatchesAnnotation')
#                         raise ValueError('Ijks')

#                     DetectionMatchesAnnotation[i[j]] = 1

#     TP = np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
#     FN = np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])

#     FP = np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])
#     F1 = 2*TP/(2*TP + FP + FN)

#     return F1, TP, FP, FN



def _F1_core_balltree(
    annos: np.ndarray, 
    boxes: np.ndarray, 
    scores: np.ndarray, 
    det_thresh: float, 
    radius: int = 25) -> Tuple[float, int, int, int]:
    """Computes F1, TP, FP, FN scores for a given set of annotations and detections.

    Args:
        annos (np.ndarray): array of bounding box coordinates in the format [xmin,ymin,xmax,ymax].
        boxes (np.ndarray): predicted bounding boxes in the format [xmin,ymin,xmax,ymax].
        scores (np.ndarray): predicted scores.
        det_thresh (float): detection threshold.
        radius (int, optional): radius of kd-tree query. Defaults to 25.

    Returns:
        Tuple[float, int, int, int]: f1, tp, fp, fn.
    """
    # filter detections by threshold 
    keep = scores > det_thresh
    boxes = boxes[keep]

    # compute scores
    scores = score_detection(ground_truth=annos, predictions=boxes, radius=radius)

    # extract scores
    TP = scores.true_positives
    FP = scores.false_positives
    FN = scores.false_negatives

    eps = 1e-12

    # compute F1
    F1 = (2 * TP) / ((2 * TP) + FP + FN + eps)

    return F1, TP, FP, FN




def _F1_core_balltree_multiclass(
    target_boxes: np.ndarray, 
    target_labels: np.ndarray, 
    pred_boxes: np.ndarray, 
    pred_scores: np.ndarray, 
    pred_labels: np.ndarray,
    det_thresh: float,
    radius: int = 25):
    """_summary_

    Args:
        target_boxes (np.ndarray): _description_
        target_labels (np.ndarray): _description_
        pred_boxes (np.ndarray): _description_
        pred_scores (np.ndarray): _description_
        pred_labels (np.ndarray): _description_
        det_thresh (float): _description_
        radius (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_
    """
    f1_scores = {}

    for lbl in np.unique(pred_labels):
        lbl_annos = target_boxes[target_labels == lbl]
        lbl_boxes = pred_boxes[pred_labels == lbl]
        lbl_scores = pred_scores[pred_labels == lbl]

        f1_score, tp, fp, fn = _F1_core_balltree(
            lbl_annos,
            lbl_boxes,
            lbl_scores,
            det_thresh,
            radius=radius
            )
        
        f1_scores[int(lbl)] = {'f1_score': f1_score, 'tp': tp, 'fp': fp, 'fn': fn}

    return f1_scores
    



def _F1_core_multiclass(
    target_boxes: np.ndarray, 
    target_labels: np.ndarray,
    pred_boxes: np.ndarray, 
    pred_scores: np.ndarray, 
    pred_labels: np.ndarray,
    det_thresh: float, 
    iou_thresh: float = 0.5) -> Dict[int, Tuple[float, int, int, int]]:
    """Computes F1-Score for a given multi-class problem. 

    Args:
        target_boxes (np.ndarray): target boxes in [xmin, ymin, xmax, ymax] format.
        target_labels (np.ndarray): array of target labels.
        pred_boxes (np.ndarray): predicted boxes in [xmin, ymin, xmax, ymax] format.
        pred_scores (np.ndarray): array of predicted scores.
        pred_labels (np.ndarray): array of predicted labels.
        det_thresh (float): detection threshold.
        iou_thresh (float, optional): IoU threshold for matching. Defaults to 0.5.

    Returns:
        Dict[int, Tuple[float, int, int, int]]: Detection results per class.
    """
    f1_scores = {}

    for lbl in np.unique(pred_labels):
        lbl_annos = target_boxes[target_labels == lbl]
        lbl_boxes = pred_boxes[pred_labels == lbl]
        lbl_scores = pred_scores[pred_labels == lbl]

        f1_score, tp, fp, fn = _F1_core_iou(
            lbl_annos,
            lbl_boxes,
            lbl_scores,
            det_thresh,
            iou_thresh=0.5
            )
        
        f1_scores[int(lbl)] = {'f1_score': f1_score, 'tp': tp, 'fp': fp, 'fn': fn}

    return f1_scores




def optimize_threshold(
    dataset: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    minthres: float = 0.3,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        dataset (pd.DataFrame): _description_
        preds (Dict[str, np.ndarray]): _description_
        minthres (float, optional): _description_. Defaults to 0.3.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: _description_
    """

    F1dict = dict()

    MIN_THR = minthres

    TPd, FPd, FNd, F1d = dict(), dict(), dict(), dict()
    thresholds = np.round(np.arange(MIN_THR,0.99,0.001), decimals=3)

    print('Optimizing threshold for validation set of %d files: '%len(preds.keys()), ','.join(list(preds.keys())))

    for resfile in preds:
        boxes = np.array(preds[resfile]['boxes'])

        TP, FP, FN = 0,0,0
        TPd[resfile] = list()
        FPd[resfile] = list()
        FNd[resfile] = list()
        F1d[resfile] = list()

        if (boxes.shape[0]>0):
            score = preds[resfile]['scores']
            
            # get annotations  
            annos = dataset[['xmin', 'ymin', 'xmax', 'ymax']].loc[dataset.filename == resfile].values

            for det_thres in thresholds:
                F1,TP,FP,FN = _F1_core_balltree(annos, boxes, score, det_thres)
                TPd[resfile] += [TP]
                FPd[resfile] += [FP]
                FNd[resfile] += [FN]
                F1d[resfile] += [F1]
        else:
            for det_thres in thresholds:
                TPd[resfile] += [0]
                FPd[resfile] += [0]
                FNd[resfile] += [0]
                F1d[resfile] += [0]
            F1 = 0
            
        F1dict[resfile]=F1

    allTP = np.zeros(len(thresholds))
    allFP = np.zeros(len(thresholds))
    allFN = np.zeros(len(thresholds))
    allF1 = np.zeros(len(thresholds))
    allF1M = np.zeros(len(thresholds))

    for k in range(len(thresholds)):
        allTP[k] = np.sum([TPd[x][k] for x in preds])
        allFP[k] = np.sum([FPd[x][k] for x in preds])
        allFN[k] = np.sum([FNd[x][k] for x in preds])
        allF1[k] = 2*allTP[k] / (2*allTP[k] + allFP[k] + allFN[k])
        allF1M[k] = np.mean([F1d[x][k] for x in preds])

    print(f'Best threshold: F1={np.max(allF1):.4f}, Threshold={thresholds[np.argmax(allF1)]:.2f}')

    return thresholds[np.argmax(allF1)], np.max(allF1), allF1, thresholds



def optimize_multiclass_threshold(
        dataset: pd.DataFrame,
        preds: Dict[str, np.ndarray],
        min_thresh: float = 0.3,
        iou_thresh: float = 0.5
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        dataset (pd.DataFrame): _description_
        preds (Dict[str, np.ndarray]): _description_
        min_thresh (float, optional): _description_. Defaults to 0.3.
        iou_thresh (float, optional): _description_. Defaults to 0.5.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: _description_
    """
    MIN_THR = min_thresh

    TPd, FPd, FNd, F1d = dict(), dict(), dict(), dict()
    thresholds = np.round(np.arange(MIN_THR,0.99,0.001), decimals=3)

    print('Optimizing threshold for validation set of %d files: '%len(preds.keys()), ','.join(list(preds.keys())))

    
    for resfile in preds:
        boxes = np.array(preds[resfile]['boxes'])

        TP, FP, FN = 0,0,0
        TPd[resfile] = list()
        FPd[resfile] = list()
        FNd[resfile] = list()
        F1d[resfile] = list()

        if (boxes.shape[0] > 0):
            scores = preds[resfile]['scores']
            labels = preds[resfile]['labels']
            
            # get annotations  
            target_boxes = dataset[['xmin', 'ymin', 'xmax', 'ymax']].loc[dataset.filename == resfile].values
            target_labels = dataset['label'].loc[dataset.filename == resfile].values

            for det_thres in thresholds:
                class_results = _F1_core_multiclass(
                    target_boxes=target_boxes,
                    target_labels=target_labels,
                    pred_boxes=boxes,
                    pred_scores=scores,
                    pred_labels=labels,
                    det_thresh=det_thres,
                    iou_thresh=iou_thresh
                )

                # init total scores 
                TP, FP, FN = 0, 0, 0

                eps = 1e-6

                # accumulate results for each class
                for class_id, class_scores in class_results.items():
                    TP += class_scores['tp']
                    FP += class_scores['fp']
                    FN += class_scores['fn']

                # calculate micro F1
                F1 = (2 * TP) / ((2 * TP) + FP + FN + eps)

                # collect metrics for each threshold
                TPd[resfile] += [TP]
                FPd[resfile] += [FP]
                FNd[resfile] += [FN]
                F1d[resfile] += [F1]

        else:
            for det_thres in thresholds:
                TPd[resfile] += [0]
                FPd[resfile] += [0]
                FNd[resfile] += [0]
                F1d[resfile] += [0]


    # aggregate over all thresholds
    allTP = np.zeros(len(thresholds))
    allFP = np.zeros(len(thresholds))
    allFN = np.zeros(len(thresholds))
    allF1 = np.zeros(len(thresholds))

    for k in range(len(thresholds)):
        allTP[k] = np.sum([TPd[x][k] for x in preds])
        allFP[k] = np.sum([FPd[x][k] for x in preds])
        allFN[k] = np.sum([FNd[x][k] for x in preds])
        allF1[k] = 2*allTP[k] / (2*allTP[k] + allFP[k] + allFN[k])

    print(f'Best threshold: F1={np.max(allF1):.4f}, Threshold={thresholds[np.argmax(allF1)]:.2f}')

    return thresholds[np.argmax(allF1)], np.max(allF1), allF1, thresholds   





        





class MIDOG2022Evaluation:
    def __init__(
            self, 
            gt_file: Union[str, pd.DataFrame],
            preds: Dict[str, Dict[str, Any]],
            output_file: str,
            det_thresh: float,
            split: str = 'test',
            bbox_size: int = 50,
            radius: int = 25
    ) -> None:
        self.gt_file = gt_file
        self.preds = preds
        self.output_file = output_file
        self.det_thresh = det_thresh
        self.split = split
        self.bbox_size = bbox_size
        self.radius = radius

        self.load_gt()

        self.ap = MAPbyDistance(radius=self.radius)
        self.per_tumor_ap = {tumor: MAPbyDistance(radius=self.radius) for tumor in self.tumor_cases}

    
    def load_gt(self) -> None:
        """Load ground truth annotations and case to tumor dictionary."""

        if isinstance(self.gt_file, str):
            dataset = pd.read_csv(self.gt_file)
        elif isinstance(self.gt_file, pd.DataFrame):
            dataset = self.gt_file
        else:
            raise TypeError('Dataset must be either str or pd.Dataframe. Got {}'.format(type(self.gt_file)))
        
        dataset = dataset.query('label == 1 and split == @self.split')

        gt = {}
        case_to_tumor = {}
        for case_ in dataset.filename.unique():
            coords = dataset.query('filename == @case_')[['x', 'y']].to_numpy()
            tumor = dataset.query('filename == @case_')['tumortype'].unique().item()
            gt[case_] = coords
            case_to_tumor[case_] = tumor

        self.gt = gt
        self.case_to_tumor = case_to_tumor
        self.tumor_cases = dataset['tumortype'].unique()



    @property
    def _metrics(self) -> Dict:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._case_results,
            "aggregates": self._aggregate_results,
        }  


    def score(self) -> None:
        """Computes case specific and aggregated results"""
        
        # init case specific results 
        self._case_results = {}

        for idx, case in enumerate(self.gt.keys()):
            if case not in self.preds:
                print('Warning: No prediction for file: ',case)
                continue

            # get case predictions
            case_preds = self.preds[case]

            preds_dict = [
                {'boxes': torch.tensor(case_preds['boxes'], dtype=torch.float),
                 'scores': torch.tensor(case_preds['scores'], dtype=torch.float),
                 'labels': torch.tensor([1,]*len(case_preds['labels']), dtype=torch.int)}
            ]

            # get case targets 
            case_targets = self.gt[case]
            
            bbox_radius = self.bbox_size / 2.

            target_dict = [
                {'boxes': torch.tensor([[x-bbox_radius, y-bbox_radius, x+bbox_radius, y+bbox_radius] for x, y in case_targets], dtype=torch.float),
                 'labels': torch.tensor([1,]*len(case_targets), dtype=torch.int)}
            ]

            # update ap metrics
            self.ap.update(preds_dict, target_dict)
            self.per_tumor_ap[self.case_to_tumor[case]].update(preds_dict, target_dict)

            
            # filter preds with label 0
            mask = case_preds['labels'] == 0

            # compute scores
            F1, tp, fp, fn = _F1_core_balltree(
                annos=[[x-bbox_radius, y-bbox_radius, x+bbox_radius, y+bbox_radius] for x, y in case_targets],
                boxes=case_preds['boxes'][mask],
                scores=case_preds['scores'][mask],
                det_thresh=self.det_thresh,
                radius=self.radius
            )

            self._case_results[case] = {'tp': tp, 'fp': fp, 'fn': fn}

        # compute aggregate results 
        self._aggregate_results = self.score_aggregates()


    def score_aggregates(self) -> Dict[str, float]:

        # init per tumor scores
        per_tumor = {tumor: {'tp': 0, 'fp': 0, 'fn': 0} for tumor in self.tumor_cases}

        # accumulate case specific scores
        tp, fp, fn = 0, 0, 0
        for case, scores in self._case_results.items():
            tp += scores['tp']
            fp += scores['fp']
            fn += scores['fn']

            per_tumor[self.case_to_tumor[case]]['tp']  += scores['tp']
            per_tumor[self.case_to_tumor[case]]['fp']  += scores['fp']
            per_tumor[self.case_to_tumor[case]]['fn']  += scores['fn']

        # init aggregated resutls
        aggregate_results = {}

        eps = 1e-6

        aggregate_results["precision"] = tp / (tp + fp + eps)
        aggregate_results["recall"] = tp / (tp + fn + eps)
        aggregate_results["f1_score"] = (2 * tp) / ((2 * tp) + fp + fn + eps)

        metric_values = self.ap.compute()
        aggregate_results["AP"] = metric_values['map'].tolist()

        # compute tumor specific restuls 
        for tumor in per_tumor:
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_precision"] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fp'] + eps)
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_recall"] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fn'] + eps)
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_f1"] = (2 * per_tumor[tumor]['tp'] + eps) / ((2 * per_tumor[tumor]['tp']) + per_tumor[tumor]['fp'] + per_tumor[tumor]['fn'] + eps)

            per_tumor_metric_values = self.per_tumor_ap[tumor].compute()
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_AP"] = per_tumor_metric_values['map'].tolist()

        return aggregate_results


    def save(self):
        with open(self.output_file, "w") as f:
                    f.write(json.dumps(self._metrics, cls=NpEncoder))  
    

    def evaluate(self, verbose: bool=False):
        self.score()
        self.save()

        

        
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
    def encode(self, obj):
        if isinstance(obj, dict):
            return super(NpEncoder, self).encode(self._convert_keys(obj))
        return super(NpEncoder, self).encode(obj)

    def _convert_keys(self, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if isinstance(key, np.integer):
                    new_key = int(key)
                else:
                    new_key = key
                if isinstance(value, dict):
                    new_dict[new_key] = self._convert_keys(value)
                else:
                    new_dict[new_key] = value
            return new_dict
        return obj



class LymphocyteEvaluation:
    def __init__(
            self, 
            gt_file: str, 
            output_file: str,
            preds: Dict[str, Dict[str, Any]],
            det_thresh: float,
            split: str = 'test',
            iou_thresh: float = 0.5,
            radius: int = 25
    ):
        self.gt_file = gt_file
        self.preds = preds
        self.output_file = output_file
        self.det_thresh = det_thresh
        self.split = split
        self.iou_thresh = iou_thresh
        self.radius = radius

        self.load_gt()

        self.ap = MAPbyDistance(radius=self.radius, class_metrics=True)
        self.per_tumor_ap = {tumor: MAPbyDistance(radius=self.radius, class_metrics=True) for tumor in self.tumor_cases}


    def load_gt(self):
        dataset = pd.read_csv(self.gt_file)
        dataset = dataset.query('split == @self.split')

        gt = {}
        case_to_tumor = {}
        for case_ in dataset.filename.unique():
            boxes = dataset.query('filename == @case_')[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
            labels = dataset.query('filename == @case_')['label'].to_numpy()
            tumor = dataset.query('filename == @case_')['tumor_id'].unique().item()
            gt[case_] = {'boxes': boxes, 'labels': labels}
            case_to_tumor[case_] = tumor

        self.gt = gt
        self.case_to_tumor = case_to_tumor
        self.tumor_cases = dataset['tumor_id'].unique().tolist()
        self.classes = dataset['label'].unique().tolist()


    @property
    def _metrics(self) -> Dict:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._case_results,
            "aggregates": self._aggregate_results,
        }  

    
    def score(self) -> None:
        """Computes case specific and aggregated results"""
        
        # init case specific results 
        self._case_results = {}

        for idx, case in enumerate(self.gt.keys()):
            if case not in self.preds:
                print('Warning: No prediction for file: ',case)
                continue

            # get case predictions
            case_preds = self.preds[case]

            # convert preds to tensors 
            preds_dict = {
                'boxes': torch.tensor(case_preds['boxes'], dtype=torch.float),
                'scores': torch.tensor(case_preds['scores'], dtype=torch.float),
                'labels': torch.tensor(case_preds['labels'], dtype=torch.int)
                }
            
            # get case targets 
            case_targets = self.gt[case]

            target_dict = {
                'boxes': torch.tensor(case_targets['boxes'], dtype=torch.float),
                'labels': torch.tensor(case_targets['labels'], dtype=torch.int)
                }

            # update ap metrics
            self.ap.update([preds_dict], [target_dict])
            self.per_tumor_ap[self.case_to_tumor[case]].update([preds_dict], [target_dict])

            # TODO: how to deal with detection below det_threshold? dont collect them?
            # mask = case_preds['labels'] == 1

            # compute scores
            f1_scores = _F1_core_balltree_multiclass(
                target_boxes=case_targets['boxes'],
                target_labels=case_targets['labels'],
                pred_boxes=case_preds['boxes'],
                pred_scores=case_preds['scores'],
                pred_labels=case_preds['labels'],
                det_thresh=self.det_thresh,
                radius=25
            )

            self._case_results[case] = f1_scores

        # compute aggregate results 
        self._aggregate_results = self.score_aggregates()


    def score_aggregates(self) -> Dict[str, float]:

        # init total scores 
        tp, fp, fn = 0, 0, 0

        # init class aggregated results
        per_class = {class_id: {'tp': 0, 'fp': 0, 'fn': 0} for class_id in self.classes}

        # init per tumor scores
        per_tumor = {tumor: {class_id: {'tp': 0, 'fp': 0, 'fn': 0} for class_id in self.classes} for tumor in self.tumor_cases}

        # accumulate case specific scores
        for case, scores in self._case_results.items():

            # accumulate results for each class
            for class_id, class_scores in scores.items():
                
                tp += class_scores['tp']
                fp += class_scores['fp']
                fn += class_scores['fn']

                per_class[class_id]['tp'] += class_scores['tp']
                per_class[class_id]['fp'] += class_scores['fp']
                per_class[class_id]['fn'] += class_scores['fn']

                per_tumor[self.case_to_tumor[case]][class_id]['tp'] += class_scores['tp']
                per_tumor[self.case_to_tumor[case]][class_id]['fp'] += class_scores['fp']
                per_tumor[self.case_to_tumor[case]][class_id]['fn'] += class_scores['fn']

        # init aggregated results
        aggregated_results = {}
        
        eps = 1e-6

        # calculate total micro averaged metrics 
        aggregated_results['micro_precision'] = tp / (tp + fp + eps)
        aggregated_results['micro_recall'] = tp / (tp + fn + eps)
        aggregated_results['micro_f1_score'] = (2 * tp) / ((2 * tp) + fp + fn + eps)

        # calculate map
        metrics = self.ap.compute()
        aggregated_results['mAP'] = metrics.map.item()

        # calculate class specific metrics
        for class_id, class_scores in per_class.items():
        
            tp = class_scores['tp']
            fp = class_scores['fp']
            fn = class_scores['fn']

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1_score = (2 * tp) / ((2 * tp) + fp + fn + eps)

            aggregated_results[class_id] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}

        # calculate micro average for each tumor
        for tumor, all_class_scores in per_tumor.items():

            tp, fp, fn = 0, 0, 0 
            
            for class_id, class_scores in all_class_scores.items():

                tp += class_scores['tp']
                fp += class_scores['fp']
                fn += class_scores['fn']

            aggregated_results[f"tumor_{tumor.replace(' ', '')}_precision"] =  tp / (tp + fp + eps)
            aggregated_results[f"tumor_{tumor.replace(' ', '')}_recall"] =  tp / (tp + fn + eps)
            aggregated_results[f"tumor_{tumor.replace(' ', '')}_f1_score"] =  (2 * tp) / ((2 * tp) + fp + fn + eps)


            # compute per tumor map 
            per_tumor_metrics = self.per_tumor_ap[tumor].compute()
            aggregated_results[f"tumor_{tumor.replace(' ', '')}_mAP"] = per_tumor_metrics['map'].item()

        return aggregated_results
        

    def save(self):
        with open(self.output_file, "w") as f:
                    f.write(json.dumps(self._metrics, cls=NpEncoder))  
    

    def evaluate(self, verbose: bool=False):
        self.score()
        self.save()



class MultiClassEvaluation:
    def __init__(
            self,
            gt_file: Union[str, pd.DataFrame],
            preds: Dict[str, Dict[str, Any]],
            output_file: str,
            det_thresh: float,
            iou_thresh: float = 0.5,
            radius: int = 25
    ):
        self.gt_file = gt_file
        self.preds = preds
        self.output_file = output_file
        self.det_thresh = det_thresh 
        self.iou_thresh = iou_thresh
        self.radius = radius

        self.load_gt()

        self.ap = MAPbyDistance(radius=self.radius)


    def load_gt(self) -> None:
        """Load ground truth annotations and case to tumor dictionary."""

        if isinstance(self.gt_file, str):
            dataset = pd.read_csv(self.gt_file)
        elif isinstance(self.gt_file, pd.DataFrame):
            dataset = self.gt_file
        else:
            raise TypeError('Dataset must be either str or pd.Dataframe. Got {}'.format(type(self.gt_file)))
        
        gt = {}
        for case_ in dataset.filename.unique():
            boxes = dataset.query('filename == @case_')[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
            labels = dataset.query('filename == @case_')['label'].to_numpy()
            centers = (boxes[:, [0, 2]] + boxes[:, [1, 3]]) / 2
            gt[case_] = {'boxes': boxes, 'centers': centers, 'labels': labels}

        self.gt = gt
        self.classes = dataset['label'].unique().tolist()


    @property
    def _metrics(self) -> Dict:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._case_results,
            "aggregates": self._aggregate_results,
        } 


    def score(self) -> None:
        """Computes case specific and aggregated results"""
        
        # init case specific results 
        self._case_results = {}

        for idx, case in enumerate(self.gt.keys()):
            if case not in self.preds:
                print('Warning: No prediction for file: ',case)
                continue

            # get case predictions
            case_preds = self.preds[case]

            # convert preds to tensors 
            preds_dict = {
                'boxes': torch.tensor(case_preds['boxes'], dtype=torch.float),
                'scores': torch.tensor(case_preds['scores'], dtype=torch.float),
                'labels': torch.tensor(case_preds['labels'], dtype=torch.int)
                }
            
            # get case targets 
            case_targets = self.gt[case]

            target_dict = {
                'boxes': torch.tensor(case_targets['boxes'], dtype=torch.float),
                'labels': torch.tensor(case_targets['labels'], dtype=torch.int)
                }

            # update ap metrics
            self.ap.update([preds_dict], [target_dict])


    	    # compute scores
            f1_scores = _F1_core_multiclass(
                target_boxes=case_targets['boxes'],
                target_labels=case_targets['labels'],
                pred_boxes=case_preds['boxes'],
                pred_scores=case_preds['scores'],
                pred_labels=case_preds['labels'],
                det_thresh=self.det_thresh,
                iou_thresh=self.iou_thresh
            )

            self._case_results[case] = f1_scores

        # compute aggregate results 
        self._aggregate_results = self.score_aggregates()



    def score_aggregates(self) -> Dict[str, float]:
        """Computes aggregated results"""

        # init total scores 
        tp, fp, fn = 0, 0, 0

        # init class aggregated results
        per_class = {class_id: {'tp': 0, 'fp': 0, 'fn': 0} for class_id in self.classes}

        # accumulate case specific scores
        for case, scores in self._case_results.items():

            # accumulate results for each class
            for class_id, class_scores in scores.items():
                
                tp += class_scores['tp']
                fp += class_scores['fp']
                fn += class_scores['fn']

                try:
                    per_class[int(class_id)]['tp'] += class_scores['tp']
                    per_class[int(class_id)]['fp'] += class_scores['fp']
                    per_class[int(class_id)]['fn'] += class_scores['fn']
                except KeyError: 
                    per_class.update(
                        {
                        int(class_id): {
                            'tp': class_scores['tp'],
                            'fp': class_scores['fp'],
                            'fn': class_scores['fn']
                            }
                        }
                    )


        # init aggregated results
        aggregated_results = {}
        
        eps = 1e-6

        # calculate total micro averaged metrics 
        aggregated_results['precision'] = tp / (tp + fp + eps)
        aggregated_results['recall'] = tp / (tp + fn + eps)
        aggregated_results['f1_score'] = (2 * tp) / ((2 * tp) + fp + fn + eps)

        # calculate map
        metrics = self.ap.compute()
        aggregated_results['mAP'] = metrics.map.item()

        # calculate class specific metrics
        for class_id, class_scores in per_class.items():
        
            tp = class_scores['tp']
            fp = class_scores['fp']
            fn = class_scores['fn']

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1_score = (2 * tp) / ((2 * tp) + fp + fn + eps)

            aggregated_results[int(class_id)] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}


        return aggregated_results


    def save(self):
        with open(self.output_file, "w") as f:
                    f.write(json.dumps(self._metrics, cls=NpEncoder))  
    

    def evaluate(self, verbose: bool=False):
        self.score()
        self.save()





class SingleClassEvaluation:
    def __init__(
            self,
            gt_file: Union[str, pd.DataFrame],
            preds: Dict[str, Dict[str, Any]],
            output_file: str,
            det_thresh: float,
            radius: int = 25
    ):
        self.gt_file = gt_file
        self.preds = preds
        self.output_file = output_file
        self.det_thresh = det_thresh 
        self.radius = radius

        self.load_gt()

        self.ap = MAPbyDistance(radius=self.radius)


    def load_gt(self) -> None:
        """Load ground truth annotations and case to tumor dictionary."""

        if isinstance(self.gt_file, str):
            dataset = pd.read_csv(self.gt_file)
        elif isinstance(self.gt_file, pd.DataFrame):
            dataset = self.gt_file
        else:
            raise TypeError('Dataset must be either str or pd.Dataframe. Got {}'.format(type(self.gt_file)))
        
        gt = {}
        for case_ in dataset.filename.unique():
            boxes = dataset.query('filename == @case_')[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
            labels = dataset.query('filename == @case_')['label'].to_numpy()
            centers = (boxes[:, [0, 2]] + boxes[:, [1, 3]]) / 2
            gt[case_] = {'boxes': boxes, 'centers': centers, 'labels': labels}

        self.gt = gt
        self.classes = dataset['label'].unique().tolist()


    @property
    def _metrics(self) -> Dict:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._case_results,
            "aggregates": self._aggregate_results,
        } 


    def score(self) -> None:
        """Computes case specific and aggregated results"""
        
        # init case specific results 
        self._case_results = {}

        for idx, case in enumerate(self.gt.keys()):
            if case not in self.preds:
                print('Warning: No prediction for file: ',case)
                continue

            # get case predictions
            case_preds = self.preds[case]

            # check for empty preds
            if case_preds['boxes'].shape[0] == 0:
                boxes = np.zeros((1, 4))
                scores = np.zeros(1)
                labels = np.zeros(1)
            else:
                boxes = case_preds['boxes']
                labels = case_preds['labels']
                scores = case_preds['scores']

            preds_dict = {
                'boxes': torch.tensor(boxes, dtype=torch.float),
                'scores': torch.tensor(scores, dtype=torch.float),
                'labels': torch.tensor(labels, dtype=torch.int)
                }
            
            # get case targets 
            case_targets = self.gt[case]

            target_dict = {
                'boxes': torch.tensor(case_targets['boxes'], dtype=torch.float),
                'labels': torch.tensor(case_targets['labels'], dtype=torch.int)
                }

            # update ap metrics
            self.ap.update([preds_dict], [target_dict])



    	    # compute scores
            f1_score, tp, fp, fn = _F1_core_balltree(
                annos=case_targets['boxes'],
                boxes=case_preds['boxes'],
                scores=case_preds['scores'],
                det_thresh=self.det_thresh,
                radius=self.radius
            )

            self._case_results[case] =  {'tp': tp, 'fp': fp, 'fn': fn}

        # compute aggregate results 
        self._aggregate_results = self.score_aggregates()



    def score_aggregates(self) -> Dict[str, float]:
        """Computes aggregated results"""

        # init total scores 
        tp, fp, fn = 0, 0, 0

        # accumulate case specific scores
        for case, scores in self._case_results.items():
            tp += scores['tp']
            fp += scores['fp']
            fn += scores['fn']

        # init aggregated results
        aggregated_results = {}
        
        eps = 1e-12

        # calculate total micro averaged metrics 
        aggregated_results['precision'] = tp / (tp + fp + eps)
        aggregated_results['recall'] = tp / (tp + fn + eps)
        aggregated_results['f1_score'] = (2 * tp) / ((2 * tp) + fp + fn + eps)

        # calculate map
        metrics = self.ap.compute()
        aggregated_results['AP'] = metrics.map.item()

        return aggregated_results


    def save(self):
        with open(self.output_file, "w") as f:
                    f.write(json.dumps(self._metrics, cls=NpEncoder))  
    

    def evaluate(self, verbose: bool=False):
        self.score()
        self.save()


















            