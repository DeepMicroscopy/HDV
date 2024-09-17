import math
from pathlib import Path
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import openslide

from openslide import OpenSlide
from abc import ABC, abstractmethod
from PIL import Image
from queue import Queue, Empty
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import nms as torch_nms
# from torchvision.ops import nms 
from tqdm.autonotebook import tqdm
from typing import Callable, Dict, List, Tuple, Union

from .dataset_adaptors import SlideObject
from utils.general import non_max_suppression

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

Coords = Tuple[int, int]



class Strategy(ABC):

    @abstractmethod
    def process_image(self, model: nn.Module, image: str, **kwargs) -> Dict[str, np.ndarray]:
        pass


class Yolov7_Inference(Strategy):
    def __init__(
            self, 
            model: nn.Module, 
            conf_thres: float=0.3,
            iou_thres_1: float=0.7,
            iou_thres_2: float=0.3,
            half_precision: bool = True,
            augment: bool = False
            ) -> None:
        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres_1 = iou_thres_1
        self.iou_thres_2 = iou_thres_2
        self.half_precision = half_precision
        self.augment = augment 

    def process_image(
            self, 
            image: Union[str, Path],
            batch_size: int = 8,
            patch_size: int = 512,
            overlap: float = 0.3,
            device: str = 'cuda',
            num_workers: int = 4,
            verbose: bool = False,
            wsi: bool = False
            ) -> Dict[str, np.ndarray]:       

        half = device != 'cpu' and self.half_precision
        if half:
            self.model.half()     
        
        # set eval mode and push to device 
        self.model.eval()
        self.model.to(device)

        
        # load image ds and dl
        if wsi:
            ds = WSI_InferenceDataset(slide_path=image, patch_size=patch_size, overlap=overlap, level=0)
        else:
            ds = ROI_InferenceDataset(image=image, size=patch_size, overlap=overlap)

        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=ds.collate_fn, drop_last=False, shuffle=False)

        # initilize results
        res = {'boxes': [], 'labels': [], 'scores': []}

        if verbose:
            pbar = tqdm(dl, desc='Processing image')
        else:
            pbar = dl

        # loop over batches
        for idx, (batch_images, batch_x, batch_y) in enumerate(pbar):
            with torch.inference_mode():

                # run model forward pass 
                batch_images = batch_images.half() if half else batch_images
                preds, _ = self.model(batch_images.to(device), augment=self.augment) 

                # apply nms patch-wise 
                preds = non_max_suppression(preds, conf_thres=self.conf_thres, iou_thres=self.iou_thres_1, labels=None, multi_label=True)

                # extract results
                boxes = [p[:, :4].cpu() for p in preds]
                scores = [p[:, 4].cpu() for p in preds]
                labels = [p[:, 5].cpu() for p in preds]

                # convert results from patch to slide coordinates
                for bbs, lbls, scrs, x_orig, y_orig in zip(boxes, labels, scores, batch_x, batch_y):
                    if bbs.size(0) > 0:
                        bbs += torch.tensor([x_orig, y_orig, x_orig, y_orig])
                        res['boxes'] += [bbs]
                        res['scores'] += [scrs]
                        res['labels'] += [lbls]
                    else:
                        continue

        # check for any predictions
        if len(res['boxes']) > 0:
            boxes = torch.cat(res['boxes'], dim=0)
            scores= torch.cat(res['scores'], dim=0)
            labels = torch.cat(res['labels'], dim=0)
            # final nms
            to_keep = torch_nms(boxes, scores, self.iou_thres_2)
            boxes = boxes[to_keep].numpy()
            scores = scores[to_keep].numpy()
            labels = labels[to_keep].numpy()
        else:
            boxes = np.empty((0, 4))
            scores = np.empty(0)
            labels = np.empty(0)

        # returns post-processed predictions or empty np.ndarray
        return {'boxes': boxes, 'scores': scores, 'labels': labels}



# ---------------------------------------------------------------------------------------------- 
# Region of intereset inference dataset using PIL ----------------------------------------------
# ----------------------------------------------------------------------------------------------


class ROI_InferenceDataset(Dataset):
    def __init__(
            self, 
            image: Union[str, Path],
            size: int=1280, 
            overlap: float=0.3
    ) -> None:
        self.image = Image.open(str(image)).convert('RGB')
        self.size = size
        self.overlap = overlap 

        self.coords = self.get_coords()
        

    def get_coords(self) -> List[Tuple[int, int]]:
        width, height = self.image.size
        coords = []
        for x in np.arange(0, width+1, self.size-(self.size*self.overlap)):
            for y in np.arange(0, height+1, self.size-(self.size*self.overlap)):
                # avoid black borders
                if x + self.size > width:
                    x = width - self.size
                if y + self.size > height:
                    y = height - self.size
                coords.append((int(x), int(y)))
        return coords
    

    def __len__(self) -> int:
        return len(self.coords)
    

    def __getitem__(self, index) -> Tuple[torch.Tensor, int, int]:
        x, y = self.coords[index]
        patch = np.asarray(self.image.crop((x, y, x+self.size, y+self.size)))
        patch = torch.from_numpy(patch / 255.).permute(2, 0, 1).type(torch.float32) 
        x = torch.as_tensor(x, dtype=torch.long)
        y = torch.as_tensor(y, dtype=torch.long)
        return patch, x, y
    

    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, List[int], List[int]]:
        images, x_coords, y_coords = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, x_coords, y_coords
    

# ---------------------------------------------------------------------------------------------- 
# Wholse slide image inference dataset using openslide -----------------------------------------
# ----------------------------------------------------------------------------------------------


def create_active_map(slide: OpenSlide) -> Tuple[np.ndarray, int]:
    """_summary_

    Args:
        slide (OpenSlide): _description_

    Returns:
        Tuple[np.ndarray, int]: _description_
    """
    downsamples_int = [int(x) for x in slide.level_downsamples]
    if 32 in downsamples_int:
        ds = 32
    elif 16 in downsamples_int:
        ds = 16

    # get overview image
    level = np.where(np.abs(np.array(slide.level_downsamples)-ds)<0.1)[0][0]
    overview = np.array(slide.read_region(level=level, location=(0,0), size=slide.level_dimensions[level]))
    
    # remove transparent alpha channel 
    alpha_zero_mask = (overview[:, :, 3] == 0)
    overview[alpha_zero_mask, :] = 255
    
    # OTSU
    gray = cv2.cvtColor(overview[:,:,0:3],cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # closing
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dil = cv2.dilate(thresh, kernel=elem)
    activeMap = cv2.erode(dil, kernel=elem)
    
    return activeMap, ds


class WSI_InferenceDataset(Dataset):
    def __init__(
            self,
            slide_path: Union[str, Path],
            patch_size: int = 512,
            level: int = 0,
            overlap: float = 0.3,
            tissue_thres: float = 0.1,
            transforms: Union[List[Callable], Callable] = None
            ) -> None:
        """_summary_

        Args:
            slide_path (Union[str, Path]): _description_
            patch_size (int, optional): _description_. Defaults to 1024.
            level (int, optional): _description_. Defaults to 1.
            overlap (float, optional): _description_. Defaults to 0.1.
            transforms (Union[List[Callable], Callable], optional): _description_. Defaults to None.
        """
        self.slide = openslide.open_slide(str(slide_path))
        self.patch_size = patch_size
        self.level = level
        self.overlap = overlap
        self.transforms = transforms
        self.tissue_thres = tissue_thres


        self.active_map, self.ds = self._create_active_map()
        self.coords = self._get_coords()
    

    @property 
    def target_size(self) -> Tuple[int, int]:
        """_summary_

        Returns:
            Tuple[int, int]: _description_
        """
        return self.slide.level_dimensions[self.level]
    

    @property
    def down_factor(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        return self.slide.level_downsamples[self.level]
    

    def _create_active_map(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        return create_active_map(self.slide)



    def _get_coords(self) -> List[Tuple[int, int]]:
        """_summary_
        """
        width, height = self.slide.dimensions
        down_factor = self.down_factor
        patch_size_level = self.patch_size * down_factor

        coords = []
        for y in range(0, height+1, int(patch_size_level * (1 - self.overlap))):
            for x in range(0, width+1, int(patch_size_level * (1 - self.overlap))):
                x = min(x, width - patch_size_level)
                y = min(y, height - patch_size_level)
                
                x_ds = int(np.floor(float(x) / self.ds))
                y_ds = int(np.floor(float(y) / self.ds))
                step_ds = int(np.ceil(float(patch_size_level)/self.ds))
                need_calc = np.sum(self.active_map[y_ds:y_ds+step_ds,x_ds:x_ds+step_ds] / 255.)>self.tissue_thres*step_ds*step_ds

                if not need_calc:
                    continue 

                coords.append((int(x), int(y)))

        return coords


    def __len__(self) -> int:
        """Returns the number of inference patches."""
        return len(self.coords)


    def __getitem__(self, idx) -> Tuple[Tensor, Coords]:
        x, y = self.coords[idx]
        patch = self.slide.read_region((x, y), level=self.level, size=(self.patch_size, self.patch_size)).convert('RGB')
        patch = np.array(patch)
        if self.transforms is not None:
            transformed = self.transforms(image=patch)
            patch = transformed['image']      
        patch = torch.from_numpy(patch / 255.).permute(2, 0, 1).type(torch.float32)
        return patch, x, y

        
    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, List[int], List[int]]:
        images, x_coords, y_coords = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, x_coords, y_coords


# ---------------------------------------------------------------------------------------------- 
# Image processor to handle different inference strategies--------------------------------------
# ----------------------------------------------------------------------------------------------


class ImageProcessor:
    def __init__(
            self, 
            strategy: Strategy,
            batch_size: int = 8,
            patch_size: int = 512,
            overlap: float = 0.3,
            device: str = 'cuda',
            num_workers: int = 4,
            verbose: bool=False, 
            wsi: bool=False) -> None:

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.overlap = overlap 
        self.device = device
        self.num_workers = num_workers
        self.verbose = verbose
        self.wsi = wsi
        
        self.settings = {
            'batch_size': self.batch_size,
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'device': self.device,
            'num_workers': self.num_workers,
            'verbose': self.verbose,
            'wsi': self.wsi}


        self._strategy = strategy


    @property
    def strategy(self) -> Strategy:
        return self._strategy
    
    @property
    def get_number_of_patches(self) -> int:
        pass

    def process_image(self, image: str) -> Dict[str, np.ndarray]:
        return self._strategy.process_image(image=image, **self.settings)
    