"""
Model loader module for loading and initializing the SLR model and device.
"""
import os
import torch
import numpy as np
from collections import OrderedDict

from vsign.models.slr_network import SLRModel
import vsign.utils as utils

def setup_device(device_id):
    """
    Set up the device for computation (GPU or CPU).
    
    Args:
        device_id (int): GPU device ID to use (e.g., 0). Use -1 for CPU.
    
    Returns:
        tuple: (device_util, torch_device)
    """
    if device_id >= 0 and torch.cuda.is_available():
        print(f"Using GPU device: {device_id}")
        torch_device = torch.device(f"cuda:{device_id}")
    else:
        print("Using CPU")
        device_id = -1  # Indicate CPU for GpuDataParallel utility if needed
        torch_device = torch.device("cpu")

    device_util = utils.GpuDataParallel()
    device_util.set_device(device_id)
    
    return device_util, torch_device

def get_dict_path(language, dict_path=None):
    """
    Get the path to the gloss dictionary file based on language or direct path.
    
    Args:
        language (str): Language identifier for loading the correct gloss dictionary.
        dict_path (str, optional): Path to the gloss dictionary (.npy file).
                                   Overrides default path based on language.
    
    Returns:
        str: Path to the gloss dictionary file.
    """
    if dict_path:
        return dict_path
    
    # Determine default dictionary path based on language
    if language == 'vsl_v0':
        dataset = 'VSL_V0'
    elif language == 'vsl_v1':
        dataset = 'VSL_V1'
    elif language == 'vsl_v2':
        dataset = 'VSL_V2'
    else:
        # Handle unsupported language
        raise ValueError(f"Unsupported language '{language}'. Use dict_path or add support.")
        
    # Construct default path relative to script or using a base path if needed
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dict_path = os.path.join(script_dir, 'data', 'processed', dataset, 'gloss_dict.npy')
    
    return dict_path

def load_gloss_dict(language, dict_path=None):
    """
    Load the gloss dictionary based on the language or direct path.
    
    Args:
        language (str): Language identifier for loading the correct gloss dictionary.
        dict_path (str, optional): Path to the gloss dictionary (.npy file).
                                   Overrides default path based on language.
    
    Returns:
        tuple: (gloss_dict, num_classes)
    """
    dict_path = get_dict_path(language, dict_path)
    
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Gloss dictionary not found at: {dict_path}. Please check the path or provide dict_path.")

    print(f"Loading gloss dictionary from: {dict_path}")
    gloss_dict = np.load(dict_path, allow_pickle=True).item()
    num_classes = len(gloss_dict) + 1  # Add 1 for blank token used in CTC
    
    return gloss_dict, num_classes

def load_model(model_path, language, torch_device, dict_path=None):
    """
    Load the SLR model from a checkpoint file.
    
    Args:
        model_path (str): Path to the pretrained model weights (.pt file).
        language (str): Language identifier for loading the correct gloss dictionary.
        torch_device (torch.device): The torch device to place the model on.
        dict_path (str, optional): Path to the gloss dictionary (.npy file).
                                   Overrides default path based on language.
    
    Returns:
        tuple: (model, gloss_dict)
    """
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at: {model_path}")
    
    gloss_dict, num_classes = load_gloss_dict(language, dict_path)
    
    print(f"Loading model definition...")
    model = SLRModel(num_classes=num_classes, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
                    loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0})

    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch_device)

    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
    else: state_dict = checkpoint

    state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch_device)
    model.eval()  # Set model to evaluation mode
    
    print("Model loaded successfully. Ready for inference.")
    return model, gloss_dict
