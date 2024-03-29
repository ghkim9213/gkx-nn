import torch
from torch import nn

class BetaEncodingBlock(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x


SUPPORTED_OUT_FEATURES = [32, 16, 8]
IN_FEATURES = 94

class BetaEncoder(nn.Module):
    def __init__(self, num_blocks=3, num_latents=6, batchnorm=True):
        super().__init__()
        assert num_blocks <= len(SUPPORTED_OUT_FEATURES)
        assert num_latents < min(SUPPORTED_OUT_FEATURES)
        self.num_blocks = num_blocks
        self.num_latents = num_latents
        self.batchnorm = batchnorm
        self.encoder = self._make_layer()

    def _make_layer(self):
        layers = []
        in_features = IN_FEATURES
        out_features = SUPPORTED_OUT_FEATURES.copy()[:self.num_blocks]
        while out_features:
            _out_features = out_features.pop(0)
            layers.append(BetaEncodingBlock(
                in_features=in_features,
                out_features=_out_features,
                batchnorm=self.batchnorm,
            ))
            in_features = _out_features
        layers.append(nn.Linear(_out_features, self.num_latents))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class FactorEncoder(nn.Module):
    def __init__(self, num_latents=6):
        super().__init__()
        self.num_latents = num_latents
        self.linear = nn.Linear(IN_FEATURES, num_latents)
    
    def forward(self, x):
        return self.linear(x)


class ConditionalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_blocks=3,
        num_latents=6,
        batchnorm=True,
        num_ensembles=10,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_latents = num_latents
        self.batchnorm = batchnorm
        self.num_ensembles = num_ensembles

        self.beta_encoders = nn.ModuleList([BetaEncoder(
            num_blocks=num_blocks,
            num_latents=num_latents,
            batchnorm=batchnorm
        ) for _ in range(num_ensembles)])
        self.factor_encoders = nn.ModuleList([FactorEncoder(
            num_latents=num_latents,
        ) for _ in range(num_ensembles)])
    
    def forward(self, characteristics, portfolio_returns):
        betas = torch.stack([
            beta_encoder(characteristics) for beta_encoder in self.beta_encoders
        ]).mean(0)
        factors = torch.stack([
            factor_encoder(portfolio_returns) for factor_encoder in self.factor_encoders
        ]).mean(0)
        pred = torch.sum(betas*factors, dim=1)
        return pred
