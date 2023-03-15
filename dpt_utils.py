import skimage
from PIL import Image
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/jupyter/DPT')
from dpt.models import DPTDepthModel

DPT_MODELS = {}
def get_dpt_model(device):
    rank = device.index
  
    if DPT_MODELS.get(rank) is None:
      model_path = "/home/jupyter/dpt_hybrid-midas-501f0c75.pt"
      model = DPTDepthModel(
          path=model_path,
          backbone="vitb_rn50_384",
          non_negative=True,
          enable_attention_hooks=False,
      )
      model = model.to(device)
      print("loaded DPT model.")
      DPT_MODELS[rank] = model
    return DPT_MODELS[rank]


def normalize_bchw(t):
    mins = torch.amin(t, dim=[1,2,3], keepdims=True)
    maxs = torch.amax(t, dim=[1,2,3], keepdims=True)
    
    denom = torch.clip(maxs - mins, min=1e-5)
    return (t - mins) / (denom)


def run_dpt(model, image):
    # image is (b,c,h,w) in [0, 1]
    
    image = image * 2  - 1
    # assert model.invert
    disparity = model(image)
    
    # disparity is (b,h,w)
    assert disparity.ndim == 3
    assert disparity.shape[0] == image.shape[0]
    assert disparity.shape[1] == image.shape[2]
    assert disparity.shape[2] == image.shape[3]
    
    disparity =  disparity[:, None]
    normalized_disparity = normalize_bchw(disparity)
    return normalized_disparity

def normalize_meanstd(tensor):
    mean = tensor.mean(axis=[1,2,3], keepdims=True)
    std = tensor.std(axis=[1,2,3], keepdims=True)
    std = torch.clip(std, min=1e-5)
    return (tensor - mean) / std


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
