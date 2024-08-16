
from dataclasses import dataclass, field
from pathlib import Path
from numpy.random import choice, randint
from typing import Callable, Dict, List, Tuple, Union
from PIL import ImageFile
import pandas as pd
import numpy as np
import openslide
import torch
import albumentations as A

from openslide import OpenSlide
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh


Coords = Tuple[int, int]


@dataclass
class SlideObject:
    slide_path: Union[str, Path]
    annotations: pd.DataFrame = None
    size: Union[int, float] = 512
    level: Union[int, float] = 0

    slide: OpenSlide = field(init=False, repr=False)

    def __post_init__(self):
        self.slide = openslide.open_slide(str(self.slide_path))


    @property
    def patch_size(self) -> Coords:
        return (self.size, self.size)
    

    @property
    def slide_size(self) -> Coords:
        return self.slide.level_dimensions[self.level]
    
    

    def load_labels(self, coords: Coords=None, label: int=None, delta: int=25) -> np.ndarray:
        """Returns annotations for a given set of coordinates. Transforms the slide annotations to patch coordinates.

        Args:
            coords (Coords): Top left patch coordinates. 
            label (int, optional): Annotations to return. Defaults to None.
            delta (int, optional): Delta to ensure that all cells are well covered by patch coordinates. Defaults to 25.

        Returns:
            np.ndarray: Boxes in patch coordinates and the labels [label, cx, cy, w, h]. 
        """
        assert self.annotations is not None, 'No annotations available.'
        assert isinstance(self.annotations, pd.DataFrame), f'Annotations must be of type pd.DataFrame, but found {type(self.annotations)}.'
        assert pd.Series(['cx', 'cy', 'w', 'h']).isin(self.annotations.columns).all(), f'DataFrame must have columns cx, cy, w, h.'
        
        # get labels
        if label is not None:
            labels = self.annotations.query('label == @label')[['label', 'cx', 'cy', 'w', 'h']]
        else:
            labels = self.annotations[['label', 'cx', 'cy', 'w', 'h']]

        # empty image 
        if labels.shape[0] == 0: 
            return np.zeros((1, 5), dtype=np.float32)
        
        # return all labels 
        if coords is None:
            labels = labels.to_numpy()

             # normalize to match yolo format
            labels[:, 1:] /= self.size 

            return labels

        else:
            # filter annotations by coordinates
            x, y = coords
            mask = ((x+delta) < labels.cx) & (labels.cx < (x+self.size-delta)) & \
                ((y+delta) < labels.cy) & (labels.cy < (y+self.size-delta))

            # create [label, cx, cy, w, h]
            patch_labels = labels[mask].to_numpy()

            if patch_labels.shape[0] == 0:
                return np.zeros((1, 5), dtype=np.float32)
    
            # convert cx, cy to patch coordinates
            patch_labels[:, 1] -= x
            patch_labels[:, 2] -= y

            # normalize to match yolo format
            patch_labels[:, 1:] /= self.size

        return patch_labels

    
    def load_image(self, coords: Coords) -> np.ndarray:
        """Returns a patch of the slide at the given coordinates."""
        patch = self.slide.read_region(location=coords, level=self.level, size=self.patch_size).convert('RGB')
        return np.array(patch)
    

class MidogDataset(Dataset):

    def __init__(
            self,
            split: str, 
            img_dir: Union[Path, str], 
            dataset: Union[pd.DataFrame, str, Path], 
            augment: bool = False, 
            box_format: str = 'xyxy',
            label_col: str = 'label', 
            domain_col: str = 'tumortype',
            filename_col: str = 'filename',
            num_samples: int = 1024,
            fg_label: int = 1,
            fg_prob: float = 0.5,
            arb_prob: float = 0.25,
            patch_size: int = 1024, 
            level: int = 0,
            radius: int = 25,
            means: List[float] = None,
            stds: List[float] = None) -> None:
        

        allowed_box_formats = ("xyxy", "cxcy")
        if box_format not in allowed_box_formats:
                raise ValueError(f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}")
        
        self.split = split
        self.box_format = box_format
        self.img_dir = Path(img_dir)
        self.dataset = dataset
        self.augment = augment
        self.label_col = label_col
        self.domain_col = domain_col
        self.filename_col = filename_col
        self.num_samples = num_samples
        self.fg_label = fg_label
        self.fg_prob =  fg_prob
        self.arb_prob = arb_prob
        self.patch_size = patch_size
        self.radius = radius
        self.means = means
        self.stds = stds 

        self._init_slide_objects()
        self._sample_coords()


    def _init_slide_objects(self) -> None:
        """Initializes SlideObjects from given dataset.

        Raises:
            ValueError: If filename_col does not exist.
            ValueError: If label_col does not exist.
            ValueError: If box_format is xyxy and columns [xmin, ymin, xmax, ymax] do not exist.
            ValueError: If box_format is cxcy and columns [x, y] do not exist.
        """
        if not isinstance(self.dataset, pd.DataFrame):
            self.dataset = pd.read_csv(self.dataset)

        if self.filename_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.filename_col}' with filenames (e.g. '012.tiff', '234.tiff') does not exist.")
        
        if self.label_col not in self.dataset.columns:
            raise ValueError(f"Column '{self.label_col}' with labels (e.g. 'Tumor cell'=1 and 'Background cell'=2) does not exist.")
        
        if self.box_format == "xyxy":
            if not pd.Series(['xmin', 'ymin', 'xmax', 'ymax']).isin(self.dataset.columns).all():
                raise ValueError(f"DataFrame expected to have columns ('xmin', 'ymin', 'xmax', 'ymax').")
            
        elif self.box_format == "cxcy":
            if not pd.Series(['x', 'y']).isin(self.dataset.columns).all():
                 raise ValueError(f"DataFrame expected to have columns ('x', 'y').")   

        if self.domain_col is not None:
            if self.domain_col not in self.dataset.columns:
                raise ValueError(f"Column '{self.domain_col}' with domain information (e.g. 'HNSCC', 'GC') does not exist.")   

        if self.split not in  ['train', 'val', 'test']:
            raise ValueError(f"Unrecognized split: {self.split}.")
        
        # filter data split
        self.dataset = self.dataset.query('split == @self.split')

        # transform boxes
        if self.box_format == 'cxcy':
            self.dataset = self.dataset.assign(cx=self.dataset['x'])
            self.dataset = self.dataset.assign(cy=self.dataset['y'])
            self.dataset = self.dataset.assign(w=self.radius * 2)
            self.dataset = self.dataset.assign(h=self.radius * 2)

        if self.box_format == 'xyxy':
            self.dataset = self.dataset.assign(cx=(self.dataset['xmin'] + self.dataset['xmax'])/2)
            self.dataset = self.dataset.assign(cy=(self.dataset['ymin'] + self.dataset['ymax'])/2)
            self.dataset = self.dataset.assign(w=self.dataset['xmax'] - self.dataset['xmin'])
            self.dataset = self.dataset.assign(h=self.dataset['ymax'] - self.dataset['ymin'])

        # make midog annotations zero-indexed
        self.dataset['label'] = self.dataset['label'] - 1

        # select columns to get data
        columns = [self.filename_col, self.label_col, 'cx', 'cy', 'w', 'h']

        if self.domain_col is not None:
            columns.extend([self.domain_col])
            domains = list(self.dataset[self.domain_col].unique())
            self.domains = domains
            self.idx_to_domain = {idx: d for idx, d in enumerate(domains)}

        # get unique filenames 
        fns = self.dataset[self.filename_col].unique().tolist()

        # initialize slideobjects
        slide_objects = {}
        for fn in tqdm(fns, desc='Initializing slide objects'):
            slide_path = self.img_dir.joinpath(fn)
            annos = self.dataset.query('filename == @fn')[columns] 
            slide_objects[fn] = SlideObject(
                slide_path=slide_path,
                annotations=annos,
                size=self.patch_size
                )
            
        # store slideobjects and dataset information
        self.slide_objects = slide_objects
        self.filenames = fns
        self.labels = [s.load_labels(label=0) for s in self.slide_objects.values()]  # collect only MF labels
        self.shapes = np.array([(self.patch_size, self.patch_size) for _ in range(len(fns))], dtype=np.float64)




    def sample_func(self, fn: str, fg_prob: float=None, arb_prob: float=None) -> Dict[str, Coords]:
            """Method that samples patch coordinates from a slide."""

            # set sampling probabilities
            fg_prob = self.fg_prob if fg_prob is None else fg_prob
            arb_prob = self.arb_prob if arb_prob is None else arb_prob

            # get slide object
            sl = self.slide_objects[fn]

            # get dims
            slide_width, slide_height = sl.slide_size
            patch_width, patch_height = sl.patch_size

            # create sampling probabilites
            sample_prob = np.array([self.arb_prob, self.fg_prob, 1-self.fg_prob-self.arb_prob])

            # sample case from probabilites (0 = random, 1 = fg, 2 = imposter)
            case = choice(3, p=sample_prob)

            # sample center coordinates
            if case == 0:       
                # random patch 
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)

            elif case == 1:     
                # filter foreground cases
                mask = sl.annotations[self.label_col] == 0

                if np.count_nonzero(mask) == 0:
                    # no annotations available -> random patch
                    x = randint(patch_width / 2, slide_width-patch_width / 2)
                    y = randint(patch_height / 2, slide_height-patch_height / 2)
                else:       
                    # get annotations
                    annos = sl.annotations[['cx', 'cy']][mask]

                    # sample foreground class
                    idx = randint(annos.shape[0])
                    x, y = annos.iloc[idx]


            elif case == 2:
                # sample imposter
                mask = sl.annotations[self.label_col] == 1

                if np.count_nonzero(mask) == 0:
                    # no imposter available -> random patch
                    x = randint(patch_width / 2, slide_width-patch_width / 2)
                    y = randint(patch_height / 2, slide_height-patch_height / 2)

                else:
                    # get annotations
                    annos = sl.annotations[['cx', 'cy']][mask]

                    # sample foreground class
                    idx = randint(annos.shape[0])
                    x, y = annos.iloc[idx]

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


            return {'file': fn, 'coords': (x, y)}
    



    def _sample_coords(self) -> None:
        """Method that samples coordinates from all slides."""

        if self.domain_col is None:
            slides = choice(self.filenames, size=self.num_samples, replace=True)
        
        else:
            # compute equal probabilities for each slide based on total number of slides per tumor type
            N = len(self.slide_objects)
            domains, counts = np.unique([v.annotations[self.domain_col].unique() for v in self.slide_objects.values()], return_counts=True)
            weights = N / counts
            weights = np.array([weights[domains == v.annotations[self.domain_col].unique()] for v in self.slide_objects.values()])
            weights = (weights / weights.sum()).reshape(-1)
            
            # sample slides
            slides = choice(self.filenames, size=self.num_samples, p=weights, replace=True)

        # sample coordinates
        coords = {}
        for idx, slide in enumerate(slides):
            coords[idx] = self.sample_func(slide, self.fg_prob, self.arb_prob)

        self.samples = coords


    def get_new_coords(self) -> None:
        """Method to get new coordinates for the next epoch."""
        self._sample_coords()



    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx) -> Tuple[Tensor, Dict[str, Tensor]]:
         # get sample
        sample = self.samples[idx]

        # extract file and coords
        file, coords = sample['file'], sample['coords']

        # get slide object
        slide = self.slide_objects[file]

        # load image and boxes
        img = slide.load_image(coords)
        targets = slide.load_labels(coords, label=0)  # load only MF labels
        labels = targets[:, 0]
        boxes = targets[:, 1:].astype(np.float32)

        if self.augment:
            if self.split == 'train':
                transforms = self.train_transform

            # if boxes are present, augment all 
            if boxes.any():
                transformed = transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = np.array(transformed['bboxes'])
                labels = np.array(transformed['class_labels'])
                if len(boxes) == 0:
                    boxes = np.zeros((1, 4) , dtype=np.float32)
                    labels = np.zeros((1,), dtype=np.int64)
            # augment only the image
            else:
                transformed = transforms(image=img, bboxes=np.zeros((0, 4)), class_labels=np.zeros(0))
                img = transformed['image']
                boxes = np.zeros((1, 4) , dtype=np.float32)
                labels = np.zeros((1,), dtype=np.int64)
        
        else:
            if len(boxes) == 0:
                boxes = np.zeros((1, 4) , dtype=np.float32)
                labels = np.zeros((1,), dtype=np.int64)

        # convert to tensor
        img = torch.from_numpy(img / 255).permute(2, 0, 1).type(torch.float32) 

        # to tensor 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64).unsqueeze(1)

    
        # create target 
        targets = torch.zeros((len(labels), 6))
        targets[:, 1:] = torch.cat([labels, boxes], dim=1)   # add extra dim to add image_idx later 

        # print(labels.shape, boxes.shape, targets.shape)


        return img, targets, slide.slide_path.name, ((self.patch_size, self.patch_size), ((1.0, 1.0), (0.0, 0.0)))
    
    
    @property
    def normalize_transform(self) -> Callable:
        """Normalization transformation. Uses Midog statistics if none are provided."""
        if self.means is None:
            # mean = [0.485, 0.456, 0.406]
            mean = [0.70860135, 0.44592966, 0.70743753]  # midog stats
        else:
            mean = self.means

        if self.stds is None:
            # std = [0.229, 0.224, 0.225]
            std = [0.05940354, 0.0838068 , 0.04765218]   # midog stats
        else:
            std = self.stds

        return A.Normalize(mean=mean, 
                        std=std,
                        max_pixel_value=255.,
                        always_apply=True)

    @property
    def train_transform(self) -> List[Callable]:
        return A.Compose([
            A.Flip(p=0.5),
            A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.1),
            A.RandomRotate90(p=0.5),
            # self.normalize_transform,
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels'])
        )


    @property
    def valid_transform(self) -> Callable:
        return self.normalize_transform
    

    def unnormalize(self, img) -> np.ndarray:
        pass 


    @staticmethod
    def collate_fn(batch):
        """Collate function for the data loader."""
        images, targets, files, shapes = zip(*batch)
        
        for i, l in enumerate(targets):      # add target image index for build_targets()
            l[:, 0] = i          
            
        # convert to tensors
        images = torch.stack(images, dim=0)
        targets = torch.cat(targets, dim=0)

        return images, targets, files, shapes
    


    def show_sample(self, idx):
        pass 