import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50


class Baseline(nn.Module):
    def __init__(self):
        super(resnet50).__init__()
        self.model = resnet50()


if __name__ == "__main__":
    model = resnet50()
    newmodel = torch.nn.Sequential(*list(model.children())[:-1])
    x = torch.rand(5, 3)
    print(model)
