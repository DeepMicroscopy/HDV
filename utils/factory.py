import yaml
import torch
import pickle
import numpy as np

from typing import Any, Dict, List, Tuple, Optional
from tqdm.notebook import tqdm
from dataclasses import InitVar, dataclass
from models.yolo import Model
from utils.torch_utils import intersect_dicts


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)



def make_yolov7_model(
        cfg: str,
        num_classes: int = 1,
        weights: str = None
) -> Model:
    """ Creates a Yolov7 model and loads a checkpoint.

    Args:
        cfg (str): Path to model configuration.
        nc (int, optional): Number of classes. Defaults to 1.
        weights (str, optional): Trained model checkpoint. Defaults to None.

    Returns:
        Model: Yolov7 model.
    """
    # load device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # init model 
    model = Model(cfg, ch=3, nc=num_classes, anchors=None).to(device)

    if weights is not None:
        ckpt = torch.load(weights, map_location=device)
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
        model.load_state_dict(state_dict, strict=False)  # load

    return model 





@dataclass(kw_only=True)
class ModelConfig:
    model_name: str 
    detector: str 
    num_classes: int
    cfg: str 
    weights: str 
    det_thresh: float 


    def update(self, new: Dict[str, Any]) -> None:
        """_summary_

        Args:
            new (Dict[str, Any]): _description_
        """
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def save(self, filepath: str) -> None:
        """_summary_

        Args:
            filepath (str): _description_
        """
        with open(filepath, 'w') as file:
            yaml.dump(self.__dict__, file)


    @classmethod
    def load(cls, filepath: str):
        """_summary_

        Args:
            filepath (str): _description_

        Returns:
            _type_: _description_
        """
        with open(filepath, 'r') as file:
            config_dict = yaml.load(file, Loader=yaml.SafeLoader)
        return cls(**config_dict)
    


@dataclass(kw_only=True)
class Yolov7_Config(ModelConfig):
    pass 





CONFIG_MAPPING = {
        'yolov7': Yolov7_Config,
        'yolov7_d6': Yolov7_Config,
    }


class ConfigCreator:

    @staticmethod
    def create(settings: Dict[str, Any]) -> ModelConfig:
        """_summary_

        Args:
            settings (Dict[str, Any]): _description_

        Raises:
            ValueError: _description_

        Returns:
            ModelConfig: _description_
        """
        detector = settings['detector'].lower()

        if detector not in CONFIG_MAPPING:
            raise ValueError(f"Model {detector} not supported.")
        
        return CONFIG_MAPPING[detector](**settings)


    @staticmethod
    def load(filepath: str) -> ModelConfig:
        """_summary_

        Args:
            filepath (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            ModelConfig: _description_
        """
        for name, config in CONFIG_MAPPING.items():
            if name in filepath:
                return config.load(filepath)
        raise ValueError(f"Model {filepath} not recognized.")
            


MODEL_MAPPINGS = {
        'yolov7': make_yolov7_model, 
        'yolov7_d6': make_yolov7_model
    }


class ModelFactory:

    @staticmethod
    def create(
            model_name: str,
            model_kwargs: Dict[str, Any] = None
            ) -> Model:
        """_summary_

        Args:
            model_kwargs (Dict[str, Any]): _description_. 
            module_kwargs (Dict[str, Any], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            BaseDetectionModule: _description_
        """    
        if model_name not in MODEL_MAPPINGS:
            raise ValueError(f"Model {model_name} not recognized.")
        
        if model_kwargs is None:
            model_kwargs = {}

        # init model 
        return MODEL_MAPPINGS[model_name](**model_kwargs)

    

    @staticmethod
    def load(
            config: ModelConfig, 
            det_thresh: float=None) -> torch.nn.Module:
        """_summary_

        Args:
            config (ModelConfig): _description_
            det_thresh (float, optional): _description_. Defaults to None.

        Returns:
            torch.nn.Module: _description_
        """
        
        # set detection threshold
        if det_thresh is None:
            det_thresh = config.det_thresh

        # get model func
        model_func = MODEL_MAPPINGS[config.detector]

        # get model settings
        model_kwargs = {
            'cfg': config.cfg,
            'weights': config.weights,
            'num_classes': config.num_classes,
        }

        # load model 
        return model_func(**model_kwargs)

        


        


        


        
        


   