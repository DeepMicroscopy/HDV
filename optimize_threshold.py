import argparse
import os
import pandas as pd
import numpy as np
import pprint

from tqdm import tqdm 
from pathlib import Path

from utils.factory import ConfigCreator, ModelFactory
from utils.inference import Yolov7_Inference, ImageProcessor
from utils.evaluation import optimize_threshold, optimize_multiclass_threshold
from utils.dataset_adaptors import (
    load_astma_df, 
    load_midog_subtyping_df, 
    load_lymph_df, 
    load_midog_df,
    load_midog_atypical_df
    )


# set default parameters
BATCH_SIZE = 8
CONFIG_FILE = None
CONFIG_PATH = 'optimized_models/'
DATASET_FILE = 'annotations/MIDOG2022_training.csv'
DETECTOR = 'yolov7'
DET_THRESH = 0.05
DEVICE = 'cuda:0'
IMG_DIR = '/data/patho/MIDOG2/'
IOU_THRESH_1 = 0.7
IOU_THRESH_2 = 0.3
MIN_THRESH = 0.2
MODEL_NAME = 'FCOS50_HNSCC'
NUM_CLASSES = 1
NUM_DOMAINS = None
NUM_WORKERS = 8
OVERLAP = 0.3
PATCH_SIZE = 1280
TUMOR_ID = None
VERBOSE = False
SPLIT = 'val'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",     type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--cfg",            type=str, help="Model configuration.")
    parser.add_argument("--config_file",    type=str, default=CONFIG_FILE, help="Existing config file.")
    parser.add_argument("--config_path",    type=str, default=CONFIG_PATH, help="Path to model configs.")
    parser.add_argument("--dataset_file",   type=str, default=DATASET_FILE, help="Dataset filepath.")
    parser.add_argument("--det_thres",      type=float, default=DET_THRESH, help="Detection threshold.")
    parser.add_argument("--detector",       type=str, default=DETECTOR, help="Model architectore.")
    parser.add_argument("--device",     	type=str, default=DEVICE, help="Device.")
    parser.add_argument("--img_dir",        type=str, default=IMG_DIR, help="Image directory.")
    parser.add_argument("--iou_thres_1",    type=float, default=IOU_THRESH_1, help="IOU threshold for patch-wise evaluation.")
    parser.add_argument("--iou_thres_2",    type=float, default=IOU_THRESH_2, help="IOU threshold for final evaluation.")
    parser.add_argument("--min_thresh",     type=float, default=MIN_THRESH, help="Minimum detection threshold.")
    parser.add_argument("--model_name",     type=str, default=MODEL_NAME, help="Model name to save config file.")
    parser.add_argument("--num_classes",    type=int, default=NUM_CLASSES, help="Number of classes.")
    parser.add_argument("--num_domains",    type=int, default=NUM_DOMAINS, help="Number of domains for DA models.")
    parser.add_argument("--num_workers",    type=int, default=NUM_WORKERS, help="Number of processes.")
    parser.add_argument("--overlap",        type=float, default=OVERLAP, help="Overlap between patches.")
    parser.add_argument("--patch_size",     type=int, default=PATCH_SIZE, help="Patch size.")
    parser.add_argument("--tumor_id",       type=str, default=TUMOR_ID, help="Which tumor type to use for optimizing threshold.")
    parser.add_argument("--verbose",        action="store_true", help="If True, prints pbar for each image.")
    parser.add_argument("--weights",        type=str, help="Path to model checkpoint.")
    parser.add_argument("--wsi",            action="store_true", help="Processes WSI")
    parser.add_argument("--split",          type=str, default=SPLIT, help="Data split to evaluate.")
    return parser.parse_args()


def main(args):

    if args.config_file is None:

        if not Path(args.weights).exists():
            raise FileNotFoundError(f'Cannot find weights: {args.weights}.')
        
        if not Path(args.config_path).exists():
            Path(args.config_path).mkdir(parents=True)


        # get model configs
        settings = {
            'model_name': args.model_name,
            'detector': args.detector,
            'cfg': args.cfg,
            'weights': args.weights,
            'det_thresh': args.det_thres,
            'num_classes': args.num_classes
        }


        print('Initializing model ...', end=' ')
        # create model config
        config_file = ConfigCreator.create(settings)

    else:
        print('Initializing model ...', end=' ')
        config_file = ConfigCreator.load(args.config_file)

    # load model 
    model = ModelFactory.load(config_file)
    print('Done.')


    print('Loaded model configurations:')
    pprint.pprint(config_file)
    print()

    # set up inference strategy
    strategy = Yolov7_Inference(
        model=model, 
        conf_thres=args.det_thres,
        iou_thres_1=args.iou_thres_1,
        iou_thres_2=args.iou_thres_2
        )

    # set up image processor
    settings = {
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'device': args.device,
        'num_workers': args.num_workers,
        'verbose': args.verbose,
        'wsi': args.wsi
    }

    # create processor
    processor = ImageProcessor(strategy=strategy, **settings)
    print('Loaded inference configurations:')
    pprint.pprint(settings)
    print()

    print('Initializing data ...', end=' ')
    if 'cells' in args.dataset_file:
        # load test slide 
        valid_dataset, _, _ = load_astma_df(args.dataset_file)
    elif 'midog' in args.dataset_file.lower():
        _, valid_dataset, _ = load_midog_df(args.dataset_file)
    elif 'subtyping' in args.dataset_file.lower():
        _, valid_dataset, _ = load_midog_subtyping_df(args.dataset_file)
    elif 'atypical' in args.dataset_file.lower():
        _, valid_dataset, _ = load_midog_atypical_df(args.dataset_file)
    elif 'lymph' in args.dataset_file.lower():
        _, valid_dataset, _ = load_lymph_df(args.dataset_file)
    else:
        raise ValueError(f'Unsupported dataset file {args.dataset_file}')
    print('Done.')

    # filter specific tumor types
    if args.tumor_id is not None:
        if 'midog' in args.dataset_file.lower():
            valid_dataset = valid_dataset.query('tumortype == @args.tumor_id')
        else:
            valid_dataset = valid_dataset.query('tumor_id == @args.tumor_id')
    print('Done.')


    # collect filenames
    filenames = valid_dataset.filename.unique()

    # init preds
    preds = {}

    # loop over files
    for file in tqdm(filenames, desc='Collecting predictions'):
        
        # get image file location
        image = os.path.join(args.img_dir, file)

        # compute predictions
        res = processor.process_image(image)

        # extract results
        boxes = res['boxes']
        scores = res['scores']
        labels = res['labels']

        if boxes.shape[0] > 0:
            preds[file] = {'boxes': boxes, 'scores': scores, 'labels': labels}
        else:
            continue 

    if 'midog' in args.dataset_file.lower():
        # select MF only 
        valid_dataset = valid_dataset.query('label == 0')

        # optimize threshold
        bestThres, bestF1, allF1, allThres = optimize_threshold(
            dataset=valid_dataset,
            preds=preds,
            minthres=args.min_thresh
        )
    else:

        # optimize multiclass threshold
        bestThres, bestF1, allF1, allThres = optimize_multiclass_threshold(
            dataset=valid_dataset,
            preds=preds,
            min_thresh=args.min_thresh,
            iou_thresh=0.5
        )

    # # reduce threshold to be more sensitive on ood data
    # propThres = np.round(bestThres - bestThres * 0.1, decimals=3)
    # propF1 = allF1[np.where(allThres == propThres)].item()
    # print(f'Proposed threshold: F1={propF1:.4f}, Threshold={propThres:.2f}')

    # updating model configs
    config_file.update({'det_thresh': float(np.round(bestThres, decimals=3))})
    print(f'Updated model configs with optimized threshold: {float(np.round(bestThres, decimals=3))}')

    if args.config_file is None:
        # save model configs
        save_path = os.path.join(args.config_path, args.model_name + '.yaml')
        config_file.save(save_path)
    else:
        config_file.save(args.config_file)



if __name__ == "__main__":
    args = get_args()
    main(args)
    print('End of script.')