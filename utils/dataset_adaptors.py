from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import openslide 
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
import cv2
import torchvision
import os


from openslide import OpenSlide
from dataclasses import dataclass, field
from pathlib import Path
from scipy import sparse
from numpy.random import randint, choice
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import Tensor
from tqdm.autonotebook import tqdm
from tiatoolbox.tools.stainaugment import StainAugmentor

from utils.torch_utils import torch_distributed_zero_first



Coords = Tuple[int, int]



# ---------------------------------------------------------------------------------------------------------------------------
# Generic slide object  ----------------------------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------------------------------------------------



@dataclass
class SlideObject:
    slide_path: Union[str, Path] 
    annotations: pd.DataFrame = field(repr=False, default=None)
    domain: Union[str, int] = field(repr=False, default=None)
    size: int = field(repr=False, default=512)
    level: int = field(repr=False, default=0)

    def __post_init__(self):
        self.slide = openslide.open_slide(str(self.slide_path))

    @property
    def patch_size(self) -> Coords:
        return (self.size, self.size)
    
    @property
    def slide_size(self) -> Coords:
        return self.slide.level_dimensions[self.level]


    def load_image(self, coords: Coords) -> np.ndarray:
        """Returns a patch of the slide at the given coordinates."""
        patch = self.slide.read_region(location=coords, level=self.level, size=self.patch_size).convert('RGB')
        return np.array(patch)


    def load_labels(self, coords: Coords=None, labels: list=None, exclude_labels: list=None, delta: int=25) -> np.ndarray:
        """Returns annotations for a given set of coordinates. Transforms the slide annotations to patch coordinates.

        Args:
            coords (Coords): Top left patch coordinates. 
            labels (list, optional): Annotations to return. Defaults to None.
            exclude_labels (list, optional): Annotations to ignore. Defaults to None.
            delta (int, optional): Delta to ensure that all cells are well covered by patch coordinates. Defaults to 25.

        Returns:
            np.ndarray: Boxes in patch coordinates and the labels [label, xmin, ymin, xmax, ymax]. 
        """
        assert self.annotations is not None, 'No annotations available.'
        assert isinstance(self.annotations, pd.DataFrame), f'Annotations must be of type pd.DataFrame, but found {type(self.annotations)}.'
        assert pd.Series(['label', 'xmin', 'ymin', 'xmax', 'ymax']).isin(self.annotations.columns).all(), f'DataFrame must have columns label, xmin, ymin, xmax, ymax.'
        
        # filter annotations by labels 
        if labels is not None:
            annotations = self.annotations.query('label in @labels')[['label', 'xmin', 'ymin', 'xmax', 'ymax']]
        # return annotations except the ones with the exclude_label
        elif exclude_labels is not None:
            annotations = self.annotations.query('label not in @exclude_labels')[['label', 'xmin', 'ymin', 'xmax', 'ymax']]
        else:
            annotations = self.annotations[['label', 'xmin', 'ymin', 'xmax', 'ymax']]

        # empty image 
        if annotations.shape[0] == 0: 
            return np.zeros((1, 5), dtype=np.float32)
        
        # return all labels 
        if coords is None:
            annotations = annotations.to_numpy()
            return annotations

        else:
            # filter annotations by coordinates
            x, y = coords
            mask = ((x+delta) < annotations.xmin) & (annotations.xmax < (x+self.size-delta)) & \
                ((y+delta) < annotations.ymin) & (annotations.ymax < (y+self.size-delta))

            # create [label, x, y, w, h]
            patch_annotations = annotations[mask].to_numpy()

            if patch_annotations.shape[0] == 0:
                return np.zeros((1, 5), dtype=np.float32)
    
            # convert to patch coordinates
            patch_annotations[:, [1, 3]] -= x
            patch_annotations[:, [2, 4]] -= y


            return patch_annotations



# ---------------------------------------------------------------------------------------------------------------------------
# MIDOG dataset adaptor ----------------------------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------------------------------------------------


def create_midog_transforms() -> Callable:
        return A.Compose([
            A.Flip(p=0.5),
            # StainAugmentor(method='macenko', p=0.5), TODO: make this optional
            A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.1),
            A.RandomRotate90(p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels'])
    )


def load_midog_df(annotations_file_path: str, box_size: int=50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(annotations_file_path)

    # make midog annotations zero-indexed for YOLOv7
    df['label'] = df['label'] - 1

    # convert annotations from x, y to xmin, ymin, xmax, ymax
    df['xmin'] = df['x'] - box_size * 0.5
    df['xmax'] = df['x'] + box_size * 0.5
    df['ymin'] = df['y'] - box_size * 0.5
    df['ymax'] = df['y'] + box_size * 0.5

    # add class names
    df['class_name'] = 'imposter'
    df.loc[df['label'] == 0, 'class_name'] = 'mitotic figure'

    # create lookups
    class_id_to_label = dict(enumerate(df.class_name.unique()))
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}
    domain_id_to_label = dict(enumerate(df.tumortype.unique()))
    domain_label_to_id = {v: k for k, v in domain_id_to_label.items()}

    # add domain ids
    df['tumortype'] = df.tumortype.map(domain_label_to_id)

    # create splits 
    train_df = df.query('split == "train"')
    valid_df = df.query('split == "val"')

    lookups = {
        'class_id_to_label': class_id_to_label,
        'class_label_to_id': class_label_to_id,
        'domain_id_to_label': domain_id_to_label,
        'domain_label_to_id': domain_label_to_id
    }

    return train_df, valid_df, lookups



class MidogDatasetAdaptor(Dataset):
    def __init__(
            self,
            img_dir_path: str, 
            dataset: pd.DataFrame,
            num_samples: int = 1024,
            patch_size: int = 640,
            filename_col: str = 'filename',
            domain_col: str = 'tumortype',
            label_col: str = 'label',
            sampling_strategy: str = 'domain_based',
            fg_label: int = 0,       # mitotif figures have label 0 in yolo
            fg_prob: float = 0.5,
            arb_prob: float = 0.25,
            level: int = 0,
            split: str = None, 
            transforms: Callable = None
    ):
        self.img_dir_path = Path(img_dir_path)
        self.dataset = dataset
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.filename_col = filename_col
        self.domain_col = domain_col
        self.label_col = label_col
        self.sampling_strategy = sampling_strategy
        self.fg_label = fg_label
        self.fg_prob = fg_prob
        self.arb_prob = arb_prob
        self.level = level
        self.transforms = transforms
        self.split = split 


        if self.filename_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.filename_col}' with filenames (e.g. '012.tiff', '234.tiff') does not exist.")
        
        if self.label_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.label_col}' with labels (e.g. 1 and 2) does not exist.")
        
        if self.domain_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.domain_col}' with domain information (e.g. 'HNSCC', 'GC') does not exist.")   


        self.filenames = self.dataset[self.filename_col].unique()

        # select columns for slide objects
        columns = [self.label_col, 'xmin', 'ymin', 'xmax', 'ymax']

        # create slide objects
        slide_objects = dict()
        for idx, filename in enumerate(tqdm(self.filenames, desc='Initializing slides')):
            slide_path = self.img_dir_path / filename
            annotations = self.dataset[self.dataset[self.filename_col] == filename]
            slide_objects[idx] = SlideObject(
                slide_path=slide_path,
                annotations=annotations[columns],
                domain=annotations[self.domain_col].unique().item(),
                size=self.patch_size,
                level=self.level,
                )
        
        # store slide objects
        self.slide_objects = slide_objects

        # create initial training samples 
        self.create_samples()


    @property
    def labels(self):
        return [s.load_labels(exclude_labels=[1]) for s in self.slide_objects.values()]  # collect only MF labels

    def __len__(self) -> int:
        return len(self.samples)
    

    def __getitem__(self, sample_id) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int]]:
        # get slide and coords from sample 
        sample = self.samples[sample_id]
        slide_id, coords = sample['file'], sample['coords']

        # get slide objects
        slide = self.slide_objects[slide_id]

        # load image and targets 
        image = slide.load_image(coords)   
        labels = slide.load_labels(coords, exclude_labels=[1])         # only load mitotic figures for training

        # get image size (always the same)
        image_hw = slide.patch_size

        if not labels.any():  # empty patch 
            xyxy_bboxes = np.array([])
            class_ids = np.array([])
        else:
            xyxy_bboxes = labels[:, 1:]
            class_ids = labels[:, 0]

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, slide.slide_path.name



    def create_samples(self) -> None:
        """Method to create training samples for one pseudo epoch."""

        # get slide ids 
        slide_ids = self.sample_slides()

        # get training samples 
        samples = dict()
        for sample_id, slide_id in enumerate(slide_ids):
            samples[sample_id] = self.sample_coords(slide_id, self.fg_prob, self.arb_prob)

        # store samples 
        self.samples = samples 


    def create_new_samples(self) -> None:
        """Wrapper method to create new samples during training."""
        self.create_samples()
        if self.split is not None:
            print(f'Created new {self.split} samples!')
        else:
            print(f'Created new samples!')


    def _sample_with_equal_probability(self) -> np.ndarray:
        """Sampling strategy that samples with equal probability from all slides."""
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, replace=True)
        return slide_ids
    

    def _sample_based_on_slides_per_domain(self) -> np.ndarray:
        """Sampling strategy that samples with equal probabilities based on total number of slides per domain."""
        assert self.domain_col is not None, 'domain_col needs to be available for this sampling strategy.'
        N = len(self.slide_objects)
        domains, counts = np.unique([v.domain for v in self.slide_objects.values()], return_counts=True)
        weights = N / counts
        weights = np.array([weights[domains == v.domain] for v in self.slide_objects.values()])
        weights = (weights / weights.sum()).reshape(-1)
        
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, p=weights, replace=True)
        return slide_ids 


    def sample_slides(self) -> np.ndarray:
        """Method to sample slide ids."""
        if self.sampling_strategy == 'default':
            return self._sample_with_equal_probability()
        elif self.sampling_strategy == 'domain_based':
            return self._sample_based_on_slides_per_domain()
        else:
            raise ValueError(f'Unsupported sampling strategy: {self.sampling_strategy}. Use onf of [default, domain_absed]')
        

    def sample_coords(self, slide_id: int, fg_prob: float=None, arb_prob: float=None) -> Dict[int, Coords]:
        """Method to sample patch coordinates from a slide."""

        # set sampling probabilities
        fg_prob = self.fg_prob if fg_prob is None else fg_prob
        arb_prob = self.arb_prob if arb_prob is None else arb_prob

        # get slide object
        sl = self.slide_objects[slide_id]

        # get dims
        slide_width, slide_height = sl.slide_size
        patch_width, patch_height = sl.patch_size

        # create sampling probabilites 
        sample_prob = np.array([self.arb_prob, self.fg_prob, 1-self.fg_prob-self.arb_prob])

        # sample case from probabilites (0 = random, 1 = mitotic figure, 2 = imposter)
        case = choice(3, p=sample_prob)

        # sample center coordinates
        if case == 0:       
            # random patch 
            x = randint(patch_width / 2, slide_width-patch_width / 2)
            y = randint(patch_height / 2, slide_height-patch_height / 2)

        elif case == 1:     
            # filter mitotic figures
            annos = sl.load_labels(labels=[0])

            if not annos.any():
                # no annotations available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)
            else:       
                # sample foreground class
                idx = randint(annos.shape[0])
                xmin, ymin, xmax, ymax = annos[idx, 1:]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2

        elif case == 2:
            # sample imposter
            annos = sl.load_labels(labels=[1])

            if not annos.any():
                # no annotations available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)
            else:
                # sample foreground class
                idx = randint(annos.shape[0])
                xmin, ymin, xmax, ymax = annos[idx, 1:]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2

        # set offsets
        offset_scale = 0.45
        xoffset = randint(-patch_width, patch_width) * offset_scale
        yoffset = randint(-patch_height, patch_height) * offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0

        return {'file': slide_id, 'coords': (x, y)}

  
# ---------------------------------------------------------------------------------------------------------------------------
# MIDOG Subtyping dataset adaptor -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------



def load_midog_subtyping_df(annotations_file_path: str, box_size: int=50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # read dataset file 
    df = pd.read_csv(annotations_file_path)

    # make midog annotations zero-indexed for YOLOv7
    df['label'] = df['label'] - 1

    # convert annotations from x, y to xmin, ymin, xmax, ymax
    df['xmin'] = df['x'] - box_size * 0.5
    df['xmax'] = df['x'] + box_size * 0.5
    df['ymin'] = df['y'] - box_size * 0.5
    df['ymax'] = df['y'] + box_size * 0.5

    # create lookups
    class_id_to_label = {
        -1: 'imposter',
        0: 'prophase',
        1: 'prometaphase',
        2: 'metaphase',
        3: 'anaphase',
        4: 'telophase',
        5: 'atypical'
    }
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}
    domain_id_to_label = dict(enumerate(df.tumortype.unique()))
    domain_label_to_id = {v: k for k, v in domain_id_to_label.items()}

    # add domain ids
    df['tumortype'] = df.tumortype.map(domain_label_to_id)

    # create splits 
    train_df = df.query('split == "train"')
    valid_df = df.query('split == "val"')

    lookups = {
        'class_id_to_label': class_id_to_label,
        'class_label_to_id': class_label_to_id,
        'domain_id_to_label': domain_id_to_label,
        'domain_label_to_id': domain_label_to_id
    }

    return train_df, valid_df, lookups



def load_midog_atypical_df(annotations_file_path: str, box_size: int=50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # read dataset file 
    df = pd.read_csv(annotations_file_path)

    # make midog annotations zero-indexed for YOLOv7
    df['label'] = df['label'] - 1

    # convert annotations from x, y to xmin, ymin, xmax, ymax
    df['xmin'] = df['x'] - box_size * 0.5
    df['xmax'] = df['x'] + box_size * 0.5
    df['ymin'] = df['y'] - box_size * 0.5
    df['ymax'] = df['y'] + box_size * 0.5

    # create lookups
    class_id_to_label = {
        -1: 'NMF',
        0: 'MF',
        1: 'AMF',
    }
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}
    domain_id_to_label = dict(enumerate(df.tumortype.unique()))
    domain_label_to_id = {v: k for k, v in domain_id_to_label.items()}

    # add domain ids
    df['tumortype'] = df.tumortype.map(domain_label_to_id)

    # create splits 
    train_df = df.query('split == "train"')
    valid_df = df.query('split == "val"')

    lookups = {
        'class_id_to_label': class_id_to_label,
        'class_label_to_id': class_label_to_id,
        'domain_id_to_label': domain_id_to_label,
        'domain_label_to_id': domain_label_to_id
    }

    return train_df, valid_df, lookups



class MidogSubtypingAdaptor(Dataset):
    def __init__(
            self,
            img_dir_path: str, 
            dataset: pd.DataFrame,
            num_samples: int = 1024,
            patch_size: int = 640,
            filename_col: str = 'filename',
            domain_col: str = 'tumortype',
            label_col: str = 'label',
            sampling_strategy: str = 'domain_based',
            level: int = 0,
            split: str = None, 
            transforms: Callable = None
    ):
        self.img_dir_path = Path(img_dir_path)
        self.dataset = dataset
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.filename_col = filename_col
        self.domain_col = domain_col
        self.label_col = label_col
        self.sampling_strategy = sampling_strategy
        self.level = level
        self.transforms = transforms
        self.split = split 


        if self.filename_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.filename_col}' with filenames (e.g. '012.tiff', '234.tiff') does not exist.")
        
        if self.label_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.label_col}' with labels (e.g. 1 and 2) does not exist.")
        
        if self.domain_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.domain_col}' with domain information (e.g. 'HNSCC', 'GC') does not exist.")   


        self.filenames = self.dataset[self.filename_col].unique()

        # select columns for slide objects
        columns = [self.label_col, 'xmin', 'ymin', 'xmax', 'ymax']

        # create slide objects
        slide_objects = dict()
        for idx, filename in enumerate(tqdm(self.filenames, desc='Initializing slides')):
            slide_path = self.img_dir_path / filename
            annotations = self.dataset[self.dataset[self.filename_col] == filename]
            slide_objects[idx] = SlideObject(
                slide_path=slide_path,
                annotations=annotations[columns],
                domain=annotations[self.domain_col].unique().item(),
                size=self.patch_size,
                level=self.level,
                )
        
        # store slide objects
        self.slide_objects = slide_objects

        # store class labels
        self.classes = self.dataset[self.label_col].unique().tolist()
        self.classes.remove(-1) # we remove imposter from the list of labels

        # create initial training samples 
        self.create_samples()


    @property
    def labels(self):
        # we ignore the imposter labels
        return [s.load_labels(exclude_labels=[-1]) for s in self.slide_objects.values()]  

    def __len__(self) -> int:
        return len(self.samples)
    

    def __getitem__(self, sample_id) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int]]:
        # get slide and coords from sample 
        sample = self.samples[sample_id]
        slide_id, coords = sample['file'], sample['coords']

        # get slide objects
        slide = self.slide_objects[slide_id]

        # load image and targets 
        image = slide.load_image(coords)   
        labels = slide.load_labels(coords, exclude_labels=[-1])         # only load mitotic figures for training

        # get image size (always the same)
        image_hw = slide.patch_size

        if not labels.any():  # empty patch 
            xyxy_bboxes = np.array([])
            class_ids = np.array([])
        else:
            xyxy_bboxes = labels[:, 1:]
            class_ids = labels[:, 0]

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, slide.slide_path.name



    def create_samples(self) -> None:
        """Method to create training samples for one pseudo epoch."""

        # get slide ids 
        slide_ids = self.sample_slides()

        # get training samples 
        samples = dict()
        for sample_id, slide_id in enumerate(slide_ids):
            samples[sample_id] = self.sample_coords(slide_id)

        # store samples 
        self.samples = samples 


    def create_new_samples(self) -> None:
        """Wrapper method to create new samples during training."""
        self.create_samples()
        if self.split is not None:
            print(f'Created new {self.split} samples!')
        else:
            print(f'Created new samples!')


    def _sample_with_equal_probability(self) -> np.ndarray:
        """Sampling strategy that samples with equal probability from all slides."""
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, replace=True)
        return slide_ids
    

    def _sample_based_on_slides_per_domain(self) -> np.ndarray:
        """Sampling strategy that samples with equal probabilities based on total number of slides per domain."""
        assert self.domain_col is not None, 'domain_col needs to be available for this sampling strategy.'
        N = len(self.slide_objects)
        domains, counts = np.unique([v.domain for v in self.slide_objects.values()], return_counts=True)
        weights = N / counts
        weights = np.array([weights[domains == v.domain] for v in self.slide_objects.values()])
        weights = (weights / weights.sum()).reshape(-1)
        
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, p=weights, replace=True)
        return slide_ids 


    def sample_slides(self) -> np.ndarray:
        """Method to sample slide ids."""
        if self.sampling_strategy == 'default':
            return self._sample_with_equal_probability()
        elif self.sampling_strategy == 'domain_based':
            return self._sample_based_on_slides_per_domain()
        else:
            raise ValueError(f'Unsupported sampling strategy: {self.sampling_strategy}. Use onf of [default, domain_absed]')
        

    def sample_coords(self, slide_id: int) -> Dict[int, Coords]:
        """Method to sample patch coordinates from a slide."""

        # set sampling probabilities
        fg_prob = 0.5
        arb_prob = 0.25

        # get slide object
        sl = self.slide_objects[slide_id]

        # get dims
        slide_width, slide_height = sl.slide_size
        patch_width, patch_height = sl.patch_size

        # create sampling probabilites 
        sample_prob = np.array([arb_prob, fg_prob, 1-fg_prob-arb_prob])

        # sample case from probabilites (0 = random, 1 = mitotic figure, 2 = imposter)
        case = choice(3, p=sample_prob)

        # sample center coordinates
        if case == 0:       
            # random patch 
            x = randint(patch_width / 2, slide_width-patch_width / 2)
            y = randint(patch_height / 2, slide_height-patch_height / 2)

        elif case == 1:     
            # we sample patches around each class with same probability
            class_idx = choice(self.classes, size=1).item()

            # get class annotations
            annos = sl.load_labels(labels=[class_idx])

            if annos.any():
                # sample foreground class
                idx = randint(annos.shape[0])
                xmin, ymin, xmax, ymax = annos[idx, 1:]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2

            else:
                # try any another foreground class
                annos = sl.load_labels(exclude_labels=[-1])

                if annos.any():
                    # sample foreground class
                    idx = randint(annos.shape[0])
                    xmin, ymin, xmax, ymax = annos[idx, 1:]
                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2

                else:
                    # no annotations available -> random patch
                    x = randint(patch_width / 2, slide_width-patch_width / 2)
                    y = randint(patch_height / 2, slide_height-patch_height / 2)

        elif case == 2:
            # sample imposter
            annos = sl.load_labels(labels=[-1])

            if annos.any():
                # sample foreground class
                idx = randint(annos.shape[0])
                xmin, ymin, xmax, ymax = annos[idx, 1:]
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2

            else:
                # no annotations available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)

        # set offsets
        offset_scale = 0.45
        xoffset = randint(-patch_width, patch_width) * offset_scale
        yoffset = randint(-patch_height, patch_height) * offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0

        return {'file': slide_id, 'coords': (x, y)}
    



# ---------------------------------------------------------------------------------------------------------------------------
# Generic Yolov7 dataset ----------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------


def convert_xyxy_to_cxcywh(bboxes):
    bboxes = bboxes.copy()
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


class Yolov7Dataset(Dataset):
    """
    A dataset which takes an object detection dataset returning (image, boxes, classes, image_id, image_hw)
    and applies the necessary preprocessing steps as required by Yolov7 models.

    By default, this class expects the image, boxes (N, 4) and classes (N,) to be numpy arrays,
    with the boxes in (x1,y1,x2,y2) format, but this behaviour can be modified by
    overriding the `load_from_dataset` method.
    """

    def __init__(
            self, 
            dataset, 
            patch_size: int = 1280,
            transforms: Callable = None):
        self.ds = dataset
        self.patch_size = patch_size
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)
    
    @property
    def labels(self):
        """Calls the dataset adaptor label property and converts to cxcywh."""
        labels = self.ds.labels
        for label in labels:
            if label.any():
                label = label.astype(float)
                label[:, 1:] = convert_xyxy_to_cxcywh(label[:, 1:].astype(float))
                label[:, 1:] /= self.patch_size  # normalized coords
        return labels

    @property
    def shapes(self):
        """Creates shapes based on patch_size."""
        return np.array([(self.patch_size, self.patch_size) for _ in range(len(self.ds))], dtype=np.float64)

    def load_from_dataset(self, index):
        """Calls the __getitem__ of the dataset adaptor"""
        image, boxes, classes, slide_path = self.ds[index]
        return image, boxes, classes, slide_path
    
    def create_new_samples(self) -> None:
        """Calls the dataset adaptor function to create new samples."""
        self.ds.create_new_samples()

    def __getitem__(self, index):
        image, boxes, classes, slide_path = self.load_from_dataset(
            index
        )

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, labels=classes)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"])
            classes = np.array(transformed["labels"])

        image = image / 255  # 0 - 1 range

        if len(boxes) != 0:
            # filter boxes with 0 area in any dimension
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_boxes]
            classes = classes[valid_boxes]

            boxes = torchvision.ops.box_convert(
                torch.as_tensor(boxes, dtype=torch.float32), "xyxy", "cxcywh"
            )
            boxes[:, [1, 3]] /= image.shape[0]  # normalized height 0-1
            boxes[:, [0, 2]] /= image.shape[1]  # normalized width 0-1
            classes = np.expand_dims(classes, 1)

            labels_out = torch.hstack(
                (
                    torch.zeros((len(boxes), 1)),
                    torch.as_tensor(classes, dtype=torch.float32),
                    boxes,
                )
            )
        else:
            labels_out = torch.zeros((1, 6))


        return (
            torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32),
            labels_out,
            slide_path,
            ((self.patch_size, self.patch_size), ((1.0, 1.0), (0.0, 0.0))),
        )

    @staticmethod
    def yolov7_collate_fn(batch):
        images, labels, files, shapes = zip(*batch)
        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets() in loss fn
        return (
            torch.stack(images, 0),
            torch.cat(labels, 0),
            files,
            shapes,
        )




# ---------------------------------------------------------------------------------------------------------------------------
# Dataloader functions ------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------


def create_midog_dataloader(
        split: str,
        batch_size: int, 
        img_dir_path: str, 
        dataset_file_path: str, 
        patch_size: int = 1280,
        num_samples: int = 1024,
        fg_prob: float = 0.5,
        arb_prob: float = 0.25,
        sampling_strategy: str = 'domain_based',
        workers: int = 8,
        world_size: int = 1,
        rank: int = -1
):
    
    with torch_distributed_zero_first(rank):
        
        # load data
        train_df, valid_df, _ = load_midog_df(dataset_file_path)

        # create midog adaptors
        if split == 'train':
            dataset = MidogDatasetAdaptor(
                split='train',
                img_dir_path=img_dir_path, 
                dataset=train_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                fg_prob=fg_prob,
                arb_prob=arb_prob,
                sampling_strategy=sampling_strategy,
                transforms=create_midog_transforms()
            )
        elif split == 'val':
            dataset = MidogDatasetAdaptor(
                split='val',
                img_dir_path=img_dir_path, 
                dataset=valid_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                fg_prob=fg_prob,
                arb_prob=arb_prob,
                sampling_strategy=sampling_strategy,
            )
        else:
            raise ValueError(f'Unrecognized split: {split}.')

        # create yolo datasets
        dataset = Yolov7Dataset(dataset, patch_size=patch_size)   

    # create dataloader 
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = DistributedSampler(dataset) if rank != -1 else None
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,  # was true
                        collate_fn=dataset.yolov7_collate_fn)
    
    return dataloader, dataset 



def create_midog_atypical_dataloader(
        split: str,
        batch_size: int, 
        img_dir_path: str, 
        dataset_file_path: str, 
        patch_size: int = 1280,
        num_samples: int = 1024,
        sampling_strategy: str = 'domain_based',
        workers: int = 8,
        world_size: int = 1,
        rank: int = -1
):
    
    with torch_distributed_zero_first(rank):
        
        # load data
        train_df, valid_df, _ = load_midog_atypical_df(dataset_file_path)

        # create midog adaptors
        if split == 'train':
            dataset = MidogSubtypingAdaptor(
                split='train',
                img_dir_path=img_dir_path, 
                dataset=train_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                sampling_strategy=sampling_strategy,
                transforms=create_midog_transforms()
            )
        elif split == 'val':
            dataset = MidogSubtypingAdaptor(
                split='val',
                img_dir_path=img_dir_path, 
                dataset=valid_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                sampling_strategy=sampling_strategy,
            )
        else:
            raise ValueError(f'Unrecognized split: {split}.')

        # create yolo datasets
        dataset = Yolov7Dataset(dataset, patch_size=patch_size)   

    # create dataloader 
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = DistributedSampler(dataset) if rank != -1 else None
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,  # was true
                        collate_fn=dataset.yolov7_collate_fn)
    
    return dataloader, dataset 



def create_midog_subtyping_dataloader(
        split: str,
        batch_size: int, 
        img_dir_path: str, 
        dataset_file_path: str, 
        patch_size: int = 1280,
        num_samples: int = 1024,
        sampling_strategy: str = 'domain_based',
        workers: int = 8,
        world_size: int = 1,
        rank: int = -1
):
    
    with torch_distributed_zero_first(rank):
        
        # load data
        train_df, valid_df, _ = load_midog_subtyping_df(dataset_file_path)

        # create midog adaptors
        if split == 'train':
            dataset = MidogSubtypingAdaptor(
                split='train',
                img_dir_path=img_dir_path, 
                dataset=train_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                sampling_strategy=sampling_strategy,
                transforms=create_midog_transforms()
            )
        elif split == 'val':
            dataset = MidogSubtypingAdaptor(
                split='val',
                img_dir_path=img_dir_path, 
                dataset=valid_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                sampling_strategy=sampling_strategy,
            )
        else:
            raise ValueError(f'Unrecognized split: {split}.')

        # create yolo datasets
        dataset = Yolov7Dataset(dataset, patch_size=patch_size)   

    # create dataloader 
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = DistributedSampler(dataset) if rank != -1 else None
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,  # was true
                        collate_fn=dataset.yolov7_collate_fn)
    
    return dataloader, dataset 



def create_astma_dataloader(
        split: str,
        batch_size: int, 
        img_dir_path: str, 
        dataset_file_path: str, 
        patch_size: int = 1280,
        num_samples: int = 1024,
        workers: int = 8,
        world_size: int = 1,
        rank: int = -1       
):
    
    with torch_distributed_zero_first(rank):
        
        # load data
        train_df, _, _ = load_astma_df(dataset_file_path)

        # create midog adaptors
        if split == 'train':
            dataset = AstmaDatasetAdaptor(
                split='train',
                img_dir_path=img_dir_path, 
                dataset=train_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                transforms=create_midog_transforms()
            )
        elif split == 'val':
            dataset = AstmaDatasetAdaptor(
                split='val',
                img_dir_path=img_dir_path, 
                dataset=train_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
            )
        else:
            raise ValueError(f'Unrecognized split: {split}.')

        # create yolo datasets
        dataset = Yolov7Dataset(dataset, patch_size=patch_size)   

    # create dataloader 
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = DistributedSampler(dataset) if rank != -1 else None
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,  # was true
                        collate_fn=dataset.yolov7_collate_fn)
    
    return dataloader, dataset 
    

def create_lymph_dataloader(
        split: str,
        batch_size: int, 
        img_dir_path: str, 
        dataset_file_path: str, 
        patch_size: int = 1280,
        num_samples: int = 1024,
        workers: int = 8,
        world_size: int = 1,
        rank: int = -1       
):
    
    with torch_distributed_zero_first(rank):
        
        # load data
        train_df, valid_df, _ = load_lymph_df(dataset_file_path)

        # create midog adaptors
        if split == 'train':
            dataset = LymphDatasetAdaptor(
                split='train',
                img_dir_path=img_dir_path, 
                dataset=train_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
                transforms=create_midog_transforms()
            )
        elif split == 'val':
            dataset = LymphDatasetAdaptor(
                split='val',
                img_dir_path=img_dir_path, 
                dataset=valid_df, 
                num_samples=num_samples, 
                patch_size=patch_size,
            )
        else:
            raise ValueError(f'Unrecognized split: {split}.')

        # create yolo datasets
        dataset = Yolov7Dataset(dataset, patch_size=patch_size)   

    # create dataloader 
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = DistributedSampler(dataset) if rank != -1 else None
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,  # was true
                        collate_fn=dataset.yolov7_collate_fn)
    
    return dataloader, dataset 


# ---------------------------------------------------------------------------------------------------------------------------
# Astma dataset adaptor -----------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------


def load_astma_df(annotations_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(annotations_file_path)

    # drop unused labels
    df = df.drop(df.query('label in ["Schrott", "Weitere"]').index)

    # add class id and class name
    class_label_to_id = {label: label_id for label_id, label in enumerate(df.label.unique())}
    class_id_to_label = {label_id: label for label, label_id in class_label_to_id.items()}

    # rename columns
    df['label'] = df.label.map(class_label_to_id)
    df['class_names'] = df.label.map(class_id_to_label)
    df = df.rename({'x1': 'xmin', 'y1': 'ymin', 'x2': 'xmax', 'y2': 'ymax'}, axis=1)

    # we select only the fully annotated slides for now
    train_slide = 'BAL Promyk Spray 4.svs'
    test_slide = 'BAL AIA Blickfang Luft.svs'

    # create splits
    train_df = df.query('filename == @train_slide')
    test_df = df.query('filename == @test_slide')

    lookups = {
        'class_id_to_label': class_id_to_label,
        'class_label_to_id': class_label_to_id
    }

    return train_df, test_df, lookups


class AstmaDatasetAdaptor(Dataset):
    def __init__(
            self,
            img_dir_path: str, 
            dataset: pd.DataFrame,
            num_samples: int = 1024,
            patch_size: int = 640,
            filename_col: str = 'filename',
            label_col: str = 'label',
            split: str = None,
            level: int = 0,
            transforms: Callable = None
    ):
        self.img_dir_path = Path(img_dir_path)
        self.dataset = dataset
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.filename_col = filename_col
        self.label_col = label_col
        self.level = level
        self.transforms = transforms
        self.split = split 

        self.filenames = self.dataset[self.filename_col].unique()

        # select columns for slide objects
        columns = [self.label_col, 'xmin', 'ymin', 'xmax', 'ymax']

        # create slide objects
        slide_objects = dict()
        for idx, filename in enumerate(tqdm(self.filenames, desc='Initializing slides')):
            slide_path = self.img_dir_path / filename
            annotations = self.dataset[self.dataset[self.filename_col] == filename]
            slide_objects[idx] = SlideObject(
                slide_path=slide_path,
                annotations=annotations[columns],
                size=self.patch_size,
                level=self.level,
                )
        
        # store slide objects
        self.slide_objects = slide_objects
        self.classes = self.dataset[self.label_col].unique().tolist()

        # create initial training samples 
        self.create_samples()


    @property
    def labels(self):
        return [s.load_labels() for s in self.slide_objects.values()]  # collect all labels 


    def __len__(self) -> int:
        return len(self.samples)
    
    
    def create_samples(self) -> None:
        """Method to create training samples for one pseudo epoch."""

        # get slide ids 
        slide_ids = self._sample_with_equal_probability()

        # get training samples 
        samples = dict()
        for sample_id, slide_id in enumerate(slide_ids):
            samples[sample_id] = self.sample_coords(slide_id)

        # store samples 
        self.samples = samples 


    def create_new_samples(self) -> None:
        """Wrapper method to create new samples during training."""
        self.create_samples()
        if self.split is not None:
            print(f'Created new {self.split} samples!')
        else:
            print(f'Created new samples!')


    def _sample_with_equal_probability(self) -> np.ndarray:
        """Sampling strategy that samples with equal probability from all slides."""
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, replace=True)
        return slide_ids
    

    def sample_coords(self, slide_id: int) -> Dict[int, Coords]:
        """Method to sample patch coordinates from a slide."""
        
        # get slide object
        sl = self.slide_objects[slide_id]

        # get dims
        slide_width, slide_height = sl.slide_size
        patch_width, patch_height = sl.patch_size

        # we use the upper half of the image for training and the lower half for evaluation
        border = slide_height // 2
        
        # we sample patches around each class with same probability
        class_idx = choice(self.classes, size=1).item()

        # sample from upper half
        if self.split == 'train':
            labels = sl.annotations[sl.annotations['ymax'] < (border - patch_height/2)]
        # sample from lower half 
        elif self.split == 'val':
            labels = sl.annotations[sl.annotations['ymin'] > (border + patch_height/2)]
        # sample from entire image
        else:
            labels = sl.annotations

        # get class annotations
        class_samples = labels.query('label == @class_idx')[['label', 'xmin', 'ymin', 'xmax', 'ymax']].to_numpy()

        # if no class annotations are present in that area get a random patch
        if class_samples.shape[0] == 0:
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                if self.split == 'train':
                    y = randint(patch_height / 2, border - patch_height / 2)
                elif self.split == 'val':
                    y = randint(border + patch_height / 2, slide_height - patch_height / 2)
                else: 
                    y = randint(patch_height / 2, slide_height - patch_height / 2)
        # sample class annotation
        else:
            idx = randint(class_samples.shape[0])
            xmin, ymin, xmax, ymax = class_samples[idx, 1:]
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2


        # set offsets
        offset_scale = 0.5
        xoffset = randint(-patch_width, patch_width) * offset_scale
        yoffset = randint(-patch_height, patch_height) * offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0

        return {'file': slide_id, 'coords': (x, y)}
    

    def __getitem__(self, sample_id) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int]]:
        # get slide and coords from sample 
        sample = self.samples[sample_id]
        slide_id, coords = sample['file'], sample['coords']

        # get slide objects
        slide = self.slide_objects[slide_id]

        # load image and targets 
        image = slide.load_image(coords)   
        labels = slide.load_labels(coords)       

        # get image size (always the same)
        image_hw = slide.patch_size

        if not labels.any():  # empty patch 
            xyxy_bboxes = np.array([])
            class_ids = np.array([])
        else:
            xyxy_bboxes = labels[:, 1:]
            class_ids = labels[:, 0]

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, slide.slide_path.name
    


            



# ---------------------------------------------------------------------------------------------------------------------------
# Lymphocite dataset adaptor ------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------



def load_lymph_df(annotations_file_path: str):
    df = pd.read_csv(annotations_file_path)

    # add class id and class name 
    class_id_to_label = {0: 'Other cell', 1: 'Tumor cell', 2: 'CD3+ positive immunce cell'}
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    # rename columns 
    df['class_names'] = df.label.map(class_id_to_label)
    df = df.rename({'x1': 'xmin', 'y1': 'ymin', 'x2': 'xmax', 'y2': 'ymax'}, axis=1)

    # create splits 
    train_df = df.query('split == "train" and tumor_id == "HNSCC"')
    valid_df = df.query('split == "val"')

    lookups = {
        'class_id_to_label': class_id_to_label,
        'class_label_to_id': class_label_to_id
    }

    return train_df, valid_df, lookups


class LymphDatasetAdaptor(Dataset):
    def __init__(
            self,
            img_dir_path: str, 
            dataset: pd.DataFrame,
            num_samples: int = 1024,
            patch_size: int = 640,
            filename_col: str = 'filename',
            label_col: str = 'label',
            split: str = None,
            level: int = 0,
            transforms: Callable = None
    ):
        self.img_dir_path = Path(img_dir_path)
        self.dataset = dataset
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.filename_col = filename_col
        self.label_col = label_col
        self.level = level
        self.transforms = transforms
        self.split = split 

        self.filenames = self.dataset[self.filename_col].unique()

        # select columns for slide objects
        columns = [self.label_col, 'xmin', 'ymin', 'xmax', 'ymax']

        # create slide objects
        slide_objects = dict()
        for idx, filename in enumerate(tqdm(self.filenames, desc='Initializing slides')):
            slide_path = self.img_dir_path / filename
            annotations = self.dataset[self.dataset[self.filename_col] == filename]
            slide_objects[idx] = SlideObject(
                slide_path=slide_path,
                annotations=annotations[columns],
                size=self.patch_size,
                level=self.level,
                )
        
        # store slide objects
        self.slide_objects = slide_objects
        self.classes = self.dataset[self.label_col].unique().tolist()

        # create initial training samples 
        self.create_samples()


    @property
    def labels(self):
        return [s.load_labels() for s in self.slide_objects.values()]  # collect all labels 


    def __len__(self) -> int:
        return len(self.samples)
    
    
    def create_samples(self) -> None:
        """Method to create training samples for one pseudo epoch."""

        # get slide ids 
        slide_ids = self._sample_with_equal_probability()

        # get training samples 
        samples = dict()
        for sample_id, slide_id in enumerate(slide_ids):
            samples[sample_id] = self.sample_coords(slide_id)

        # store samples 
        self.samples = samples 


    def create_new_samples(self) -> None:
        """Wrapper method to create new samples during training."""
        self.create_samples()
        if self.split is not None:
            print(f'Created new {self.split} samples!')
        else:
            print(f'Created new samples!')


    def _sample_with_equal_probability(self) -> np.ndarray:
        """Sampling strategy that samples with equal probability from all slides."""
        slide_ids = choice(list(self.slide_objects.keys()), size=self.num_samples, replace=True)
        return slide_ids
    

    def sample_coords(self, slide_id: int) -> Dict[int, Coords]:
        """Method to sample patch coordinates from a slide."""
        
        # get slide object
        sl = self.slide_objects[slide_id]

        # get dims
        slide_width, slide_height = sl.slide_size
        patch_width, patch_height = sl.patch_size

        # we use the upper half of the image for training and the lower half for evaluation
        border = slide_height // 2
        
        # we sample patches around each class with same probability
        class_idx = choice(self.classes, size=1).item()

        # get class annotations
        annos = sl.load_labels(labels=[class_idx])

        if annos.any():
            # sample foreground class
            idx = randint(annos.shape[0])
            xmin, ymin, xmax, ymax = annos[idx, 1:]
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2

        else:
            # no annotations available -> random patch
            x = randint(patch_width / 2, slide_width-patch_width / 2)
            y = randint(patch_height / 2, slide_height-patch_height / 2)


        # set offsets
        offset_scale = 0.5
        xoffset = randint(-patch_width, patch_width) * offset_scale
        yoffset = randint(-patch_height, patch_height) * offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0

        return {'file': slide_id, 'coords': (x, y)}
    


    def __getitem__(self, sample_id) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int]]:
        # get slide and coords from sample 
        sample = self.samples[sample_id]
        slide_id, coords = sample['file'], sample['coords']

        # get slide objects
        slide = self.slide_objects[slide_id]

        # load image and targets 
        image = slide.load_image(coords)   
        labels = slide.load_labels(coords)       

        # get image size (always the same)
        image_hw = slide.patch_size

        if not labels.any():  # empty patch 
            xyxy_bboxes = np.array([])
            class_ids = np.array([])
        else:
            xyxy_bboxes = labels[:, 1:]
            class_ids = labels[:, 0]

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, slide.slide_path.name