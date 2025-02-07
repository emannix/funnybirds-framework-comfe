import torch.nn as nn
from abc import abstractmethod
from pdb import set_trace as pb
import torch
import numpy as np

class ModelExplainerWrapper:

    def __init__(self, model, explainer):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions 
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer    
        """
        self.model = model
        self.explainer = explainer

    def predict(self, input):
        return self.model.forward(input)

    def explain(self, input):
        return self.explainer.explain(self.model, input)


class AbstractModel(nn.Module):
    def __init__(self, model):
        """
        An abstract wrapper for PyTorch models implementing functions required for evaluation.
        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, input):
        return self.model

class StandardModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class ViTModel(AbstractModel):
    """
    A wrapper for ViT models.
    Args:
        model: PyTorch neural network model
    """

    def forward(self, input):
        input = nn.functional.interpolate(input, (224,224)) # ViT expects input of size 224x224
        return self.model(input)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

# ==========================================================


class ComFeModel(nn.Module):

    def __init__(self, model, explainer):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions 
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer    
        """

        super().__init__()
        self.model = model
        self.explainer = explainer

    def forward(self, input):
        # from torchvision.transforms import functional as F
        # F.to_pil_image(input[0]).show()
        z_patch = self.model.backbone(input)
        # ========================================
        paint_map, paint, paint_cover = self.model.base_clustering.forward_pred(z_patch)

        # rescramble classes
        rescramble = np.arange(50).astype(str)
        rescramble.sort()
        rescramble = rescramble.astype(int).argsort()
        paint = paint[:, :-1][:, rescramble]
        paint_map = paint_map[:,:,:,rescramble]
        # ========================================

        return paint

    def predict(self, input):
        return self.model.forward(input)

    def explain(self, input):
        z_patch = self.model.backbone(input)
        # ========================================
        paint_map, paint, paint_cover = self.model.base_clustering.forward_pred(z_patch)

        rescramble = np.arange(50).astype(str)
        rescramble.sort()
        rescramble = rescramble.astype(int).argsort()
        paint = paint[:, :-1][:, rescramble]
        paint_map = paint_map[:,:,:,rescramble]
        # ========================================

        pred = paint.argmax(dim=1)
        paint_map_idx = paint_map[torch.arange(paint.shape[0]),:,:, pred]

        paint_map_idx = nn.functional.interpolate(paint_map_idx[:, None, :, :], (256,256), mode='bilinear')
        paint_map_idx = paint_map_idx.squeeze()

        return paint_map_idx








