from cellpose import models, core
import torch
from PIL import Image
import numpy as np


def get_cellpose_sam_output(imgs):

    gpu = False
    if torch.cuda.is_available():
        gpu = True

    imgs = imgs.to_list()

    model = models.CellposeModel(gpu=gpu)

    preds, flows, styles = model.eval(imgs)

    return preds