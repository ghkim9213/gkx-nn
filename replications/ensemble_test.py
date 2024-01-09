import torch
from torch import nn
from gkx_nn.models import ConditionalAutoEncoderEnsemble

bsz = 8
n_features = 94
n_channels = 10
input0 = torch.randn(bsz, n_features)
input1 = torch.randn(bsz, n_features)
model = ConditionalAutoEncoderEnsemble()
model(input0, input1)
import pdb; pdb.set_trace()
