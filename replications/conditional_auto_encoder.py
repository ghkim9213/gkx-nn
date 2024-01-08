from tqdm import tqdm

import optuna
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
ROOT_DIR = "./data"
BATCH_SIZE = 10000
ds_factory = GKXDatasetFactory(root_dir=ROOT_DIR)
ds_factory.download_data()
tvtsets = ds_factory.split_by_year(
    split_ratio=[.3, .2, .5],
    from_year="2011",
    to_year="2020",
    scaling_func=normalize,
)
trainloader, validloader, testloader = (DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in tvtsets)
input_extractor = lambda loaded: loaded[:2]
label_extractor = lambda loaded: loaded[2]

# model
model = ConditionalAutoEncoder(
    batchnorm=True # Algorithm 7, GKX2020
).double()

# trainer
# constant configs
SEED = 0
MAX_EPOCH = 10
PATIENCE = 5

# hp space
def suggest_hp(trial):
    return (
        trial.suggest_categorical("lr", [1e-3, 1e-2]),
        trial.suggest_float("l1", 1e-5, 1e-3, log=True),
    )

def objective(trial):
    best_valid_loss = float("Inf")
    patience = 0
    lr, l1 = suggest_hp(trial)
    print(f"[Trial {trial.number}] lr: {lr:>7f}, l1: {l1:>7f}")
    with mlflow.start_run():
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = LassoLoss(model.named_parameters(), l1=l1)
        for epoch in range(MAX_EPOCH):
            torch.manual_seed(SEED)
            print(f"[Trial {trial.number} / epoch {epoch}] training...")
            train_loss = train_epoch(
                model, device, trainloader, optimizer, loss_fn,
                input_extractor=input_extractor,
                label_extractor=label_extractor,
            )
            print(f"[Trial {trial.number} / epoch {epoch}] validating...")
            valid_loss = validate_epoch(
                model, device, validloader, loss_fn,
                input_extractor=input_extractor,
                label_extractor=label_extractor,
            )
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
            else:
                patience += 1
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            print(f"[Trial {trial.number} / epoch {epoch}] train_loss {train_loss:>7f}, valid_loss {valid_loss:7f}")
            if patience == PATIENCE:
                print(f"[Trial {trial.number} / epoch {epoch}] early stopped")
                break
    return best_valid_loss

study = optuna.create_study(study_name="init_train", direction="minimize")
study.optimize(objective, n_trials=5)