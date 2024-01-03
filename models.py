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
    def __init__(self, num_blocks=3, num_betas=6, batchnorm=True):
        super().__init__()
        assert num_blocks <= len(SUPPORTED_OUT_FEATURES)
        assert num_betas < min(SUPPORTED_OUT_FEATURES)
        self.num_blocks = num_blocks
        self.num_betas = num_betas
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
        layers.append(nn.Linear(_out_features, self.num_betas))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class FactorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear
