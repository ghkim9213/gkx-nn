from tqdm import tqdm

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from gkx_nn.data import GKXDatasetFactory, normalize, minmax
from gkx_nn.experiment import train_epoch, validate_epoch
from gkx_nn.models import ConditionalAutoEncoder
from gkx_nn.loss_functions import LassoLoss

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
ROOT_DIR = "../data"
BATCH_SIZE = 2**11 # bsz = 10000 for 30 years, GKX 2020
ds_factory = GKXDatasetFactory(root_dir=ROOT_DIR)
ds_factory.download_data()
datasets = ds_factory.split_by_year(
    split_ratio=[1.],
    from_year="2001",
    to_year="2005",
    scaling_func=normalize,
)
trainloader, validloader = (DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in datasets)
input_extractor = lambda loaded: loaded[:2]
label_extractor = lambda loaded: loaded[2]

# model
NUM_ENSEMBLES = 10
model = ConditionalAutoEncoder(
    batchnorm=True,  # batchnorm, Algorithm 7, GKX2020
    num_ensembles=NUM_ENSEMBLES, # num_ensembles=10, GXK2020
)

# trainer
SEED = 0
NUM_EPOCHS = 3 # set by hpo
L1 = 4e-5 # set by hpo
LR = 1e-3 # set by hpo

patience = 0
prev_valid_loss = float("inf")
with mlflow.start_run():
    mlflow.log_params({"lr": LR, "l1": L1})
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = LassoLoss(model.named_parameters(), l1=L1)
    for epoch in range(NUM_EPOCHS):
        torch.manual_seed(SEED)
        print(f"[Epoch {epoch}] training...")
        train_loss = train_epoch(
            model, device, trainloader, optimizer, loss_fn,
            input_extractor=input_extractor,
            label_extractor=label_extractor,
        )
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        print(f"[Epoch {epoch}] train_loss {train_loss:>7f}")
    mlflow.pytorch.log_model(model, f"model")