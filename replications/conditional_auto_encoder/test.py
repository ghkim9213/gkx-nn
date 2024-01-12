import os
import mlflow
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from gkx_nn.data import GKXDatasetFactory, normalize
from gkx_nn.models import ConditionalAutoEncoder
from gkx_nn.experiment import test_epoch

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
ROOT_DIR = "../data"
BATCH_SIZE = 2**11
ds_factory = GKXDatasetFactory(root_dir=ROOT_DIR)
testset = ds_factory.split_by_year(
    split_ratio=[1.],
    from_year="2011",
    to_year="2020",
    scaling_func=normalize,
)[0]
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
input_extractor = lambda loaded: loaded[:2]
label_extractor = lambda loaded: loaded[2]

# model
client = MlflowClient()
run = client.search_runs("0")[0]
model_uri = os.path.join(run.info.artifact_uri, "model")
model = mlflow.pytorch.load_model(model_uri)

# test
accuracy = test_epoch(
    model=model,
    device=device,
    dataloader=testloader,
    input_extractor=input_extractor,
    label_extractor=label_extractor,
)
import pdb; pdb.set_trace()