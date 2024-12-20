import argparse
import os
import pandas as pd
import numpy as np 
import pprint
import torch 
import pickle

from pathlib import Path
from tqdm.autonotebook import tqdm 
from torch.utils.data import DataLoader

from utils.factory import ConfigCreator, ModelFactory
from utils.inference import Patch_InferenceDataset, adjust_bounding_boxes
from utils.general import non_max_suppression
from utils.evaluation import MultiClassEvaluation, SingleClassEvaluation


BATCH_SIZE = 8
CONFIG_FILE = 'optimized_models/yolov7_d6_ALL_0.yaml'
DATASET_FILE = 'annotations/lung_test_1class.csv'
DEVICE = 'cuda'
IMG_DIR = '/data/patho/biomag/lung/images'
PREDS_FILE = None
DET_THRESH = None
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
    parser.add_argument("--dataset_file",   type=str, default=DATASET_FILE, help="Dataset filepath.")
    parser.add_argument("--iou_thres_1",    type=float, default=IOU_THRESH_1, help="IOU threshold for patch-wise evaluation.")
    parser.add_argument("--iou_thres_2",    type=float, default=IOU_THRESH_2, help="IOU threshold for final evaluation.")
    parser.add_argument("--config_file",    type=str, default=CONFIG_FILE, help='Model configurations.')
    parser.add_argument("--det_thres",      type=float, default=DET_THRESH, help="Set specific detection threshold.")
    parser.add_argument("--device",     	type=str, default=DEVICE, help="Device.")
    parser.add_argument("--img_dir",        type=str, default=IMG_DIR, help="Image directory.")
    parser.add_argument("--num_workers",    type=int, default=NUM_WORKERS, help="Number of processes.")
    parser.add_argument("--output_file",    type=str, default=OUTPUT_FILE, help="Filename to save results. (Default: None) Created from Config file.")
    parser.add_argument("--overlap",        type=float, default=OVERLAP, help="Overlap between patches.")
    parser.add_argument("--patch_size",     type=int, default=PATCH_SIZE, help="Patch size.")
    parser.add_argument("--save_path",      type=str, default=SAVE_PATH, help="Directory to save results.")
    parser.add_argument("--split",          type=str, default=SPLIT, help="Data split to evaluate.")
    parser.add_argument("--verbose",        action="store_true", help="If True, prints pbar for each image.")
    parser.add_argument("--preds_file",     type=str, default=PREDS_FILE, help="Path to predictions.")
    return parser.parse_args()




def main(args):

        
    print('Initializing model ...', end=' ')
    # load model config
    config_file = ConfigCreator.load(args.config_file)

    
    # if prediction is None, run inference 
    if args.preds_file is None:

        # set specific detection threshold
        det_thresh = args.det_thres if args.det_thres is not None else config_file.det_thresh
        config_file.update({'det_thresh': det_thresh})

        # load model 
        model = ModelFactory.load(config_file, det_thresh)
        print('Done.')

        print('Loaded model configurations:')
        pprint.pprint(config_file)
        print()
        
        device = args.device
        
        # set precision
        half = device != 'cpu' 
        if half:
            model.half()     

        # set eval mode and push to device 
        model.eval()
        model.to(device)
        
    
        # set up dataset
        dataset = Patch_InferenceDataset(
            image_dir=args.img_dir,
            size=args.patch_size
        )
        
        # set up dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn
        )
        
        # init results 
        results = {}

        # loop over files
        for batch in tqdm(dataloader, desc='Collecting predictions'):
            
            # get data from batch 
            images, filenames = batch
            
            # set precision
            images = images.half() if half else images
            
            
            with torch.no_grad():
            
                # run model 
                preds, _ = model(images.to(device), augment=args.augment) 

            # apply nms patch-wise 
            preds = non_max_suppression(preds, conf_thres=config_file.det_thresh, iou_thres=args.iou_thres_1, labels=None, multi_label=True)
            
            # extract results
            boxes = [p[:, :4].cpu().numpy() for p in preds]
            scores = [p[:, 4].cpu().numpy() for p in preds]
            labels = [p[:, 5].cpu().numpy() for p in preds]
            
            # adjust boxes to original coordinates
            boxes = [adjust_bounding_boxes(dets, pad_size=128) for dets in boxes]
            
            # collect predictions
            for (filename, bxs, scrs, lbls) in zip(filenames, boxes, scores, labels):           
                
                if bxs:
                    
                    bxs = np.stack(bxs, axis=0)
                    scrs = np.array(scrs)
                    lbls = np.array(lbls)          
                    
                    results[filename] = {'boxes': bxs, 'scores': scrs, 'labels': lbls}
                    
                    if args.verbose:
                        print(f'{filename}: {bxs.shape[0]}, {scrs}')
                else:
                    
                    results[filename] = {'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])}
    else:
        results = pickle.load(open(args.preds_file, 'rb'))

    # check save path
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # init output file
    if args.output_file is not None:
        output_file = save_path.joinpath(args.output_file)    
    else:
        output_file = save_path.joinpath(config_file.model_name + '.pkl')


    if '6class' in args.dataset_file.lower():

        # load annotations 
        eval_dataset = pd.read_csv(args.dataset_file)
        eval_dataset.loc[:, 'label'] -= 1                   # make annotations 0 indexed
        eval_dataset = eval_dataset.query('label != -1')    # remove imposter labels

        # convert annotations from x, y to xmin, ymin, xmax, ymax
        eval_dataset['xmin'] = eval_dataset['x'] - 25
        eval_dataset['xmax'] = eval_dataset['x'] + 25
        eval_dataset['ymin'] = eval_dataset['y'] - 25
        eval_dataset['ymax'] = eval_dataset['y'] + 25

        evaluation = MultiClassEvaluation(
            gt_file=eval_dataset,
            preds=results,
            output_file=output_file,
            det_thresh=config_file.det_thresh,
            iou_thresh=0.5
        )
    
    elif '1class' in args.dataset_file.lower():
        
        # load annotations
        eval_dataset = pd.read_csv(args.dataset_file)

        # convert annotations from x, y to xmin, ymin, xmax, ymax
        eval_dataset['xmin'] = eval_dataset['x'] - 25
        eval_dataset['xmax'] = eval_dataset['x'] + 25
        eval_dataset['ymin'] = eval_dataset['y'] - 25
        eval_dataset['ymax'] = eval_dataset['y'] + 25

        evaluation = SingleClassEvaluation(
            gt_file=eval_dataset,
            preds=results,
            output_file=output_file,
            det_thresh=config_file.det_thresh,
            radius=25
        )

    else:
        raise ValueError(f'Unsupported dataset file')
    
        
    # evaluate
    evaluation.evaluate()
    print('Evaluation done.')
    

    print(f'Evaluation results:')
    # show aggregate metrics
    pprint.pprint(evaluation._metrics['aggregates'])
        

if __name__ == "__main__":
    args = get_args()
    main(args)
    print('End of script.')







    