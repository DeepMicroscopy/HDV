import json
import numpy as np
import os
import pandas as pd
import pickle
import torch
from pathlib import Path
import pprint
import argparse

from extract_features import get_all_annotations
from utils.hellinger import domain_specific_hellinger_distance
from utils.gdv import domain_specific_gdv
from utils.factory import ConfigCreator
from utils.dataset_adaptors import load_astma_df


SAVE_DIR= 'results/'
ONLY_BORDER = False
BOX_FORMAT = 'cxcy'
DOMAIN_COL = 'tumortype'
METRIC = 'hdv'
METHOD = 'autohist'
AGGREGATION = 'mean'
SPLIT = 'test'
C = 100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",      type=str, help='Model configurations.')
    parser.add_argument("--box_format",       type=str, default=BOX_FORMAT, help='Box format (default: cxcy).')
    parser.add_argument("--dataset_file",     type=str, help="Dataset filepath.")
    parser.add_argument("--img_dir",          type=str, help="Image directory.")
    parser.add_argument("--save_dir",         type=str, default=SAVE_DIR, help="Location to save bhatta results.")
    parser.add_argument("--feature_dir",      type=str, help="Location of features and targets.")
    parser.add_argument("--only_border",      action="store_true", help="Extracts only features from border cases.")
    parser.add_argument("--domain_col",       type=str, default=DOMAIN_COL, help="Column with different domains, e.g. tumortypes (default: tumor_id).")
    parser.add_argument("--metric",           type=str, default=METRIC, help="Metric to compute.")
    parser.add_argument("--method",           type=str, default=METHOD, help='Method to estimate histogram bins.')
    parser.add_argument("--coefficient",      action='store_false', help='Computes similarity coefficient instead of distance.')
    parser.add_argument("--aggregation",      type=str, default=AGGREGATION, help='Aggregation function.')
    parser.add_argument("--split",            type=str, default=SPLIT)
    parser.add_argument("--dimensions",       type=int, default=C, help="Number of dimensions to select with highest variance.")
    parser.add_argument("--class_metrics",    action="store_true", help="Returns between class metrics.")
    return parser.parse_args()


def main(args):

    # load model config
    config_file = ConfigCreator.load(args.config_file)

    # get model name
    model_name = config_file.model_name
    print(f'\nComputing similarities for model: {model_name}')
    print(f'Computing metric: ', args.metric)


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

    # testset labels
    test_annos = torch.tensor([v for l in test_samples.values() for v in l['labels']])
    if 'midog' in args.dataset_file.lower():
        test_annos -= 1

    # set feature dir
    feature_dir = Path(args.feature_dir)

    print('Loading features and targets ...', end=' ')
    if not feature_dir.joinpath('features_' + model_name + '.pkl').exists():
        raise FileNotFoundError(f'Features for model {model_name} not found.')
    else:
        features = pickle.load(open(feature_dir.joinpath('features_' + model_name + '.pkl'), 'rb'))

    if not feature_dir.joinpath('domains_' + model_name + '.pkl').exists():
        raise FileNotFoundError(f'Domains for model {model_name} not found.')
    else:
        domains = pickle.load(open(feature_dir.joinpath('domains_' + model_name + '.pkl'), 'rb'))
    print('Done.')

    # import pdb
    # pdb.set_trace()

    print('Computing similarities ...', end=' ')
    # compute bhatta coef
    if args.metric == 'hdv':
        all_dist = domain_specific_hellinger_distance(    
            features_dict=features, 
            domains=domains, 
            labels=test_annos, 
            codes=test_codes, 
            method=args.method, 
            distance=args.coefficient, 
            decimals=4,
            aggregation=args.aggregation,
            num_dimensions=args.dimensions,
            class_metrics=args.class_metrics)
        
    elif args.metric == 'gdv':
        all_dist = domain_specific_gdv(
            features_dict=features, 
            domains=domains, 
            labels=test_annos, 
            codes=test_codes, 
            decimals=4
        )
    else:
        ValueError(f'Metric {args.metric} not recognized.')
    print('Done.')
    
    print('\nCoefficients:')
    pprint.pprint(all_dist)

    # set results dir
    results_dir = Path(args.save_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    print('Saving results ...', end=' ')
    # save results
    result_name = results_dir.joinpath(f'{args.metric}_{model_name}.pkl')
    with open(result_name, 'wb') as file:
        pickle.dump(all_dist, file)
    print('Done.')

if __name__ == "__main__":
    args = get_args()
    main(args)
    print('End of script.')
