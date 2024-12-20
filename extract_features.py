import argparse
import gc
import pprint 
import numpy as np
import os
import pandas as pd
import pickle
import torch
import openslide 

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.autonotebook import tqdm
from typing import Dict, List, Tuple, Union

from utils.feature_extraction import (
    FeatureExtractor, 
    FeatureCollector, 
    CenteredObjectDataset, 
    extract_features
)
from utils.factory import ConfigCreator, ModelFactory
from utils.dataset_adaptors import MidogDatasetAdaptor, load_astma_df

import utils.constants as constants



def filter_coords(x, y, width, height, delta=50) -> bool:
    """Returns True if x or y are close to the border."""
    left_right_border = (x < delta) or (x > (width - delta))
    top_bottom_border = (y < delta) or (y > (height - delta))
    return left_right_border or top_bottom_border


def get_all_annotations(
        dataset: pd.DataFrame, 
        img_dir_path: Union[str, Path], 
        domain_col: str = 'tumor_id',
        box_format: str = 'cxcy', 
        only_border: bool = False
        ) -> Dict[str, List[Tuple[int, int]]]:
    """Returns all annotations from the dataset centered in the moddle of the patch."""
    img_dir_path = Path(img_dir_path)
    samples = {}
    dataset = dataset.dropna(subset=['label'])
    files = dataset.filename.unique()
    for file in files:
        slide = openslide.open_slide(str(img_dir_path.joinpath(file)))
        width, height = slide.dimensions
        subdata = dataset.query('filename == @file').copy()
        label = subdata['label']
        if box_format == 'cxcy':
            coords = subdata[['x', 'y']]
            if only_border:
                mask = coords.apply(lambda df: filter_coords(df.x, df.y, width, height), axis=1)
                coords = coords.drop(coords[~mask].index)
                label = label.drop(label[~mask].index)
            x1 = coords['x'] - 25
            y1 = coords['y'] - 25
            x2 = coords['x'] + 25
            y2 = coords['y'] + 25
            boxes = np.stack((x1, y1, x2, y2), axis=1).astype(int)
        elif box_format == 'xyxy':
            boxes = subdata[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy().astype(int)
        else:
            raise ValueError(f'Unrecognized box_format: {box_format}')
        domain = 'None' if domain_col == 'None' else subdata[domain_col].unique().item()
        samples[file] = {
            'boxes': boxes.tolist(),
            'labels': label.values.tolist(),
            'domain': domain}
    return samples



BATCH_SIZE = 8
CONFIG_FILE = 'optimized_models/yolov7_d6_ALL_0.yaml'
DATASET_FILE = 'annotations/midog_2022_test.csv'
DEVICE = 'cuda'
IMG_DIR = '/data/patho/MIDOG2/finalTest'
NUM_WORKERS = 8
PATCH_SIZE = 1280
VERBOSE = False
SAVE_PATH = '/data/jonas/midog/features'
ONLY_BORDER = False
BOX_FORMAT = 'cxcy'
DOMAIN_COL = 'tumortype'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",       type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--config_file",      type=str, default=CONFIG_FILE, help='Model configurations.')
    parser.add_argument("--dataset_file",     type=str, default=DATASET_FILE, help="Dataset filepath.")
    parser.add_argument("--device",     	  type=str, default=DEVICE, help="Device.")
    parser.add_argument("--img_dir",          type=str, default=IMG_DIR, help="Image directory.")
    parser.add_argument("--num_workers",      type=int, default=NUM_WORKERS, help="Number of processes.")    
    parser.add_argument("--patch_size",       type=int, default=PATCH_SIZE, help="Patch size.")
    parser.add_argument("--verbose",          action="store_true", help="If True, prints pbar for each image.")
    parser.add_argument("--save_path",        type=str, default=SAVE_PATH, help="Location to save features and targets.")
    parser.add_argument("--only_border",      action="store_true", help="Extracts only features from border cases.")
    parser.add_argument("--box_format",       type=str, default=BOX_FORMAT, help="Box format (default: xyxy).")
    parser.add_argument("--domain_col",       type=str, default=DOMAIN_COL, help="Column with different domains, e.g. tumortypes (default: tumor_id).")
    return parser.parse_args()



def main(args):
    
    print('Initializing model ...', end=' ')
    # load model config
    config_file = ConfigCreator.load(args.config_file)

    # load model 
    model = ModelFactory.load(config_file)
    print('Done.')

    print('Loaded model configurations:')
    pprint.pprint(config_file)
    print()

    print('Initializing data ...', end=' ')
    if 'cells' in args.dataset_file:
        # load test slide 
        _, test_dataset, _ = load_astma_df(args.dataset_file)
    elif 'midog' in args.dataset_file.lower() or 'lymph' in args.dataset_file.lower():
        dataset = pd.read_csv(args.dataset_file)
        # filter eval samples 
        test_dataset = dataset.query('split == "test"')
    else:
        raise ValueError(f'Unsupported dataset file {args.dataset_file}')
    print('Done.')
    
    # create test codes
    if args.domain_col == 'None':
        test_codes = {0: 'None'}
    else:
        test_codes = {k: v for k, v in enumerate(test_dataset[args.domain_col].unique())}

    # get test samples and labels
    test_samples = get_all_annotations(
        dataset=test_dataset, 
        img_dir_path=args.img_dir, 
        domain_col=args.domain_col, 
        only_border=args.only_border,
        box_format=args.box_format
        )

    # set up feature extraction
    if str.lower(config_file.detector) == 'yolov7':
        layers = constants.YOLO_LAYERS
    elif str.lower(config_file.detector) == 'yolov7_d6':
        layers = constants.YOLO_D6_LAYERS
    else:
        raise ValueError(f'Unrecognized model for {config_file.detector}. Cannot determine feature extraction layer.')
    
    # start feature extraction
    test_features, test_classes, test_domains = extract_features(
                                        model=model, 
                                        layer=layers,
                                        img_dir=args.img_dir,
                                        patch_size=args.patch_size,
                                        samples=test_samples,
                                        tumor_code=list(test_codes.values()),
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        verbose=args.verbose)

    print('\nExtracted feature dimensions: ')
    # print feature dimensions
    for layer, features in test_features.items():
        print(layer, features.shape)
    print()

    # set up saving of results
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # use save name from config file
    save_name = config_file.model_name

    feature_path = save_path.joinpath('features_' + save_name + '.pkl')
    domain_path = save_path.joinpath('domains_' + save_name + '.pkl')
    classes_path = save_path.joinpath('classes_' + save_name + '.pkl')


    print('Saving results ...', end=' ')
    # save features and targets 
    with open(feature_path, 'wb') as file:
        pickle.dump(test_features, file)

    with open(domain_path, 'wb') as file:
        pickle.dump(test_domains, file)

    with open(classes_path, 'wb') as file:
        pickle.dump(test_classes, file)
    print('Done.')


if __name__ == "__main__":
    args = get_args()
    main(args)
    print('End of script.')











