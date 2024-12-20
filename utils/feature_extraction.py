import gc
import math
import numpy as np
import torch
import torch.nn as nn
import os
import openslide 
from openslide import OpenSlide
from pathlib import Path

from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from typing import Callable, Dict, Iterable, List, Tuple, Union



class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: List[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {}

    
    def save_features_hook(self, layer_id: str) -> Callable:
        """Hook to save features at certain layer. Very tightly coupled to Torchvision FPN code."""
        def fn(module, input, output):
                self._features[layer_id] = output.detach().cpu()
        return fn    


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass to return a dictionary with features from each layer."""

        # add hook
        hooks = []
        modules = dict(self.model.named_modules())
        for layer in self.layers:
            if not layer in modules.keys():
                raise ValueError('layer {} not recognized'.format(layer))
            hooks.append(modules[layer].register_forward_hook(self.save_features_hook(layer)))

        # forward pass 
        with torch.no_grad():
            _ = self.model(x)

        # remove hooks
        for hook in hooks:
            hook.remove()
            
        return self._features
    


    def get_down_factors(self, batch_size: int, patch_size: int) -> Dict[str, Tuple[Tuple[int], int]]:
        """Performs one dummy forward pass to calculate the downsampling factor at each layer."""
        x = torch.randn((batch_size, 3, patch_size, patch_size))
        self.eval()
        self.to('cuda')
        with torch.no_grad():
            features = self(x.to('cuda'))
        dims = {}
        for layer_id, values in features.items():
            down_factor = patch_size // values.shape[-1]
            dims[layer_id] = down_factor
        self.clear_collection()
        return dims


    def clear_collection(self):
        """Init new collection"""
        self._features.clear()




class FeatureCollector:
    def __init__(
            self,
            downsampling_factors: Dict[str, int],
            mode: str = 'mean'
    ) -> None:
        """Class to collect object level features and target information.

        Args:

            downsampling_factors (Dict[str, int]): Dictionary with layernames and downsamling factors.
        """
        self.downsampling_factors = downsampling_factors
        self.mode = mode

        self._collection = self._init_collection()
        self._classes = []
        self._domains = []


    def _init_collection(self) -> Dict[str, Tensor]:
        """Initialize feature dictionary with empty tensors of specific dimensions."""
        collection = {}
        for layer, _ in self.downsampling_factors.items():
            collection[layer] = []
        return collection    


    @staticmethod
    def project_object(box: Tensor, down_factor: int) -> Tensor:
        """Project object onto feature map with certain downsampling factor.

        Args:
            box (Tuple[int]): Bounding box locations [x1, y1, x2, y2]
            down_factor (int): Downsampling factor.

        Returns:
            Tuple[int]: Projected object [x1, y1, x2, y2]
        """
        # project the object 
        projection = torch.floor(box / down_factor)

        # ensure that minimum is still 1x1 box 
        if not any(projection):
            projection[2:] = 1   # [0,0,1,1]

        return projection
    

    def extract_features(self, box: Tensor, features: Tensor) -> Tensor:
        """Extract object level features.

        Args:
            box (Tuple[int]): Bounding box locations [x1, y1, x2, y2]
            features (Tensor): Feature tensor [C, H, W]

        Returns:
            Tensor: Object level feature
        """
        x1, y1, x2, y2 = box.type(torch.int).tolist()
        if self.mode == 'mean':
            object_feautres = torch.mean(features[:, y1:y2, x1:x2], dim=(1,2))
        elif self.mode == 'all':
            object_feautres = features[:, y1:y2, x1:x2]
        else:
            raise ValueError(f'Unsupported feature extraction mode for {self.mode}')
        return object_feautres


    def update(
            self, 
            features: Dict[str, Tensor], 
            boxes: List[Tensor], 
            classes: Tensor, 
            domains: Tensor
            ) -> None:
        """Collect object level features and target information from a batch.

        Args:
            feautres (Dict[str, Tensor]): Dictionary with layer feature tensor {layer: [B, C, H, W]}
            boxes (Tensor): Tensor of bounding box locations [B, N, 4]
            classes (Tensor): Tensor of class labels [B, N, 1]
            domains (Tensor): Tensor of domain labels [B, N, 1]
        """
        for layer_id, features in features.items():

            # get downsampling factor
            down_factor = self.downsampling_factors[layer_id]

            # extract object level features from each image
            for image_features, image_boxes in zip(features, boxes):

                # loop over samples
                for box in image_boxes:

                    # project box onto feature map
                    projected_box = self.project_object(box, down_factor)

                    # extract object level feature
                    object_features = self.extract_features(projected_box, image_features)

                    # collect features
                    self._collection[layer_id].append(object_features)


        # update class information
        self._classes.extend(classes.tolist())
        self._domains.extend(domains.tolist())


    def aggregate_collection(self) -> None:
        """Aggregates the collected features """
        assert len(self._collection.values()) > 0, 'Feature collection is empty.'
        assert len(self._classes) > 0, 'Class information is empty.'
        assert len(self._domains) > 0, 'Domain information is empty.'
        if self.mode == 'mean':
            self.collection = {k: torch.stack(v, dim=0) for k, v in self._collection.items()}
        elif self.mode == 'all':
            self.collection = self._collection
        self.classes = torch.tensor(self._classes)
        self.domains = torch.tensor(self._domains)



    def get_collection(self) -> Tensor:
        """Returns the collected features."""
        if not hasattr(self, 'collection'):
            self.aggregate_collection()
        return self.collection
    

    def get_classes(self) -> Tensor:
        """Returns the collected classes."""
        if not hasattr(self, 'classes'):
            self.aggregate_collection()
        return self.classes
    

    def get_domains(self) -> Tensor:
        """Returns the collected classes."""
        if not hasattr(self, 'domains'):
            self.aggregate_collection()
        return self.domains


    def get_results(self) -> Tuple[Tensor]:
        """Returns the collected information."""
        if not hasattr(self, 'collection'):
            self.aggregate_collection()
        return self.collection, self.classes, self.domains
    




class CenteredObjectDataset(Dataset):
    def __init__(
        self, 
        file: str,
        patch_size: int,
        coords: List[List[int]],
        classes: List[int],
        domain: Union[str, int],
    ) -> None:
        """Class for object centered feature extraction. For each object 
        we extract one patch from the slide with the object centered in 
        the middle.

        Args:
            slide (SlideObject): SlideObject to sample patches. 
            coords (List[List[int]]): List of bounding box coordinates [N, 4]
            classes (List[int]): List of class labels [N]
            domains (Union[str, int]): Domain label [1]
        """
        self.slide = self.read_slide(file)
        self.coords = coords
        self.classes = classes
        self.domain = domain
        self.size = patch_size


    @staticmethod
    def read_slide(filename: str) -> Image.Image | OpenSlide:
        """Opens either PIL Image or OpenSlide object."""
        if Path(filename).suffix in ['.tiff', '.tif', '.jpeg', '.png']:
            return Image.open(filename).convert('RGB')
        else:
            return openslide.open_slide(filename)
        

    def load_image(self, coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Reads a patch from different image objects."""
        x1, y1, x2, y2 = coords
        if isinstance(self.slide, Image.Image):
            image = self.slide.crop((x1, y1, x2, y2)).convert('RGB')
        elif isinstance(self.slide, OpenSlide):
            width = x2 - x1
            height = y2 - y1 
            image = self.slide.read_region((x1, y1), 0, (width, height)).convert('RGB') 
        else:
            raise ValueError(f'Unsupported slide type: {type(self.slide)}')
        return np.array(image) / 255.


    def __len__(self) -> int: 
        return len(self.coords)
    
    def __getitem__(self, index) -> torch.Tensor:
        """Takes a bounding box and puts it into the center of a patch."""
        x1, y1, x2, y2 = self.coords[index]
        cls = self.classes[index]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        top_left_x = cx - self.size // 2
        top_left_y = cy - self.size // 2
        bottom_right_x = top_left_x + self.size
        bottom_right_y = top_left_y + self.size
        patch = self.load_image((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        patch = torch.from_numpy(patch).permute(2, 0, 1).type(torch.float32)
        target = torch.as_tensor((int(x1-top_left_x), int(y1-top_left_y), int(x2-top_left_x), int(y2-top_left_y)), dtype=torch.long)
        cls = torch.as_tensor(cls, dtype=torch.long)
        return patch, target, cls, self.domain
    
    @staticmethod
    def collate_fn(batch):
        images, boxes, classes, domains = zip(*batch)
        images = torch.stack(images)
        boxes = torch.stack(boxes).unsqueeze(1)
        classes = torch.stack(classes)
        return images, boxes, classes, domains
    



def extract_features(
        model: torch.nn.Module, 
        layer: str, 
        img_dir: str, 
        patch_size: int, 
        samples: Dict[str, List[Tuple[int, int]]], 
        tumor_code: List[str],
        batch_size: int=4,
        num_workers: int=4,
        verbose: bool=False) -> Tuple[torch.Tensor]:
        """Extract features from the layer of the model on a given set of samples from the sampler.

        Args:
            model (nn.Module): Pytorch model.
            layer (str): Layer name from which to extract feautres.
            samples (Dict[str, List[Tuple[int, int]]]): Dictionary with filenames and coordinates.

        Returns:
            Tuple[torch.Tensor]: Features and targets
        """

        # build featuer extractor
        feature_extractor = FeatureExtractor(model, layer)
        feature_extractor.eval()
        feature_extractor.to('cuda')

        # get downsampling factors 
        down_factors = feature_extractor.get_down_factors(batch_size, patch_size=512)

        # build feature collection
        feature_collector = FeatureCollector(down_factors)

        # start extracting patches
        with torch.no_grad():
            for file, values in tqdm(samples.items(), desc='Processing images'):

                # extract values
                coords = values['boxes']
                labels = values['labels']
                domain = values['domain']

                # get slide location
                slide = os.path.join(img_dir, file)

                # create dataset and dataloader 
                ds = CenteredObjectDataset(slide, patch_size, coords, labels, domain)
                dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, collate_fn=ds.collate_fn)

                if verbose:
                    dl = tqdm(dl, desc='Processing patches')

                # loop over the samples 
                for (images, boxes, classes, domains) in dl:

                    features = feature_extractor(images.to('cuda'))                    

                    # add target domain
                    domains = torch.tensor([tumor_code.index(d) for d in domains])

                    # collect features and targets
                    feature_collector.update(features, boxes, classes, domains)

                    # empty collection
                    feature_extractor.clear_collection()     

                    gc.collect()
                    torch.cuda.empty_cache()
        
        # collect results
        features, classes, domains = feature_collector.get_results()

        return features, classes, domains 

