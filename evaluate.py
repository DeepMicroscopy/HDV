import argparse
import os
import pandas as pd
import numpy as np 
import pprint

from pathlib import Path
from tqdm.autonotebook import tqdm 

from utils.inference import Yolov7_Inference, ImageProcessor
from utils.factory import ConfigCreator, ModelFactory
from utils.evaluation import MIDOG2022Evaluation, LymphocyteEvaluation, MultiClassEvaluation
from utils.dataset_adaptors import load_astma_df


BATCH_SIZE = 8
DEVICE = 'cuda'
DET_THRESH = 0.05
IOU_THRESH_1 = 0.7
IOU_THRESH_2 = 0.3
NUM_WORKERS = 8
OUTPUT_FILE = None
OVERLAP = 0.3
PATCH_SIZE = 1280
SAVE_PATH = 'results/'
SPLIT = 'test'
VERBOSE = False
WSI = False



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment",        action="store_true", help="Use test time augmentation.")
    parser.add_argument("--batch_size",     type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--iou_thres_1",    type=float, default=IOU_THRESH_1, help="IOU threshold for patch-wise evaluation.")
    parser.add_argument("--iou_thres_2",    type=float, default=IOU_THRESH_2, help="IOU threshold for final evaluation.")
    parser.add_argument("--config_file",    type=str, help='Model configurations.')
    parser.add_argument("--dataset_file",   type=str, help="Dataset filepath.")
    parser.add_argument("--device",     	type=str, default=DEVICE, help="Device.")
    parser.add_argument("--img_dir",        type=str, help="Image directory.")
    parser.add_argument("--num_workers",    type=int, default=NUM_WORKERS, help="Number of processes.")
    parser.add_argument("--output_file",    type=str, default=OUTPUT_FILE, help="Filename to save results. (Default: None) Created from Config file.")
    parser.add_argument("--overlap",        type=float, default=OVERLAP, help="Overlap between patches.")
    parser.add_argument("--patch_size",     type=int, default=PATCH_SIZE, help="Patch size.")
    parser.add_argument("--save_path",      type=str, default=SAVE_PATH, help="Directory to save results.")
    parser.add_argument("--split",          type=str, default=SPLIT, help="Data split to evaluate.")
    parser.add_argument("--verbose",        action="store_true", help="If True, prints pbar for each image.")
    parser.add_argument("--wsi",            action="store_true")
    parser.add_argument("--overwrite",      action="store_true", help="Overwrites existing result file.")
    return parser.parse_args()




def main(args):

    if not Path(args.config_file).exists():
        raise FileNotFoundError(f'Cannot find config file: {args.config_file}.')
    
    if not Path(args.dataset_file).exists():
        raise FileNotFoundError(f'Cannot find dataset file: {args.dataset_file}.')
    
    print('Initializing model ...', end=' ')
    # load model config
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
        conf_thres=config_file.det_thresh,
        iou_thres_1=args.iou_thres_1,
        iou_thres_2=args.iou_thres_2,
        augment=args.augment,
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
        _, eval_dataset, _ = load_astma_df(args.dataset_file)
    elif 'midog' in args.dataset_file.lower() or 'lymph' in args.dataset_file.lower():
        dataset = pd.read_csv(args.dataset_file)
        # filter eval samples 
        eval_dataset = dataset.query('split == @args.split')
    elif 'atypical' in args.dataset_file.lower():
        dataset = pd.read_csv(args.dataset_file)
        # make midog annotations zero-indexed for YOLOv7
        dataset['label'] = dataset['label'] - 1
        # filter eval samples and ignore imposter 
        eval_dataset = dataset.query('split == @args.split & label >= 0') # in our atypical dataset labels are 0:NMF, 1:MF, 2:MF
        eval_dataset['xmin'] = eval_dataset['x'] - 25
        eval_dataset['xmax'] = eval_dataset['x'] + 25
        eval_dataset['ymin'] = eval_dataset['y'] - 25
        eval_dataset['ymax'] = eval_dataset['y'] + 25
    else:
        raise ValueError(f'Unsupported dataset file {args.dataset_file}')
    print('Done.')

    # check save path
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # init output file
    if args.output_file is not None:
        output_file = save_path.joinpath(args.output_file)    
    else:
        output_file = save_path.joinpath(config_file.model_name + '.json')

    # skip if already exists
    if output_file.exists() and not args.overwrite:
        raise ValueError(f'Output file already exists. Skipping evaluation.')


    # collect filenames
    filenames = eval_dataset.filename.unique()

    # init preds
    preds = {}

    # loop over files
    for file in tqdm(filenames, desc='Collecting predictions'):
        
        # get image file location
        image = os.path.join(args.img_dir, file)

        # compute predictions
        res = processor.process_image(image)

        # collect predictions
        preds[file] = res


    print('Starting evaluation ...')
    if 'midog' in args.dataset_file.lower():
        evaluation = MIDOG2022Evaluation(
            gt_file=args.dataset_file,
            output_file=output_file,
            preds=preds,
            det_thresh=config_file.det_thresh,
            split=args.split
        )
    elif 'cells' in args.dataset_file.lower() or 'atypical' in args.dataset_file.lower():
        evaluation = MultiClassEvaluation(
            gt_file=eval_dataset,
            preds=preds,
            output_file=output_file,
            det_thresh=config_file.det_thresh,
            iou_thresh=0.5
        )

    elif 'lymph' in args.dataset_file.lower():
        evaluation = LymphocyteEvaluation(
            gt_file=args.dataset_file,
            output_file=output_file,
            preds=preds,
            det_thresh=config_file.det_thresh,
            split=args.split
        )

    else:
        raise ValueError(f'Unrecognized dataset: {args.dataset_file}')

    # evaluate
    evaluation.evaluate()
    print('Evaluation done.')
    

    print(f'Evaluation results for {args.split} split')
    # show aggregate metrics
    pprint.pprint(evaluation._metrics['aggregates'])


if __name__ == "__main__":
    args = get_args()
    main(args)
    print('End of script.')







    