import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.data import GKXDatasetFactory
from ..utils.models import ConditionalAutoEncoder

# setup
ROOT_DIR = "../data"

ds_factory = GKXDatasetFactory(root_dir=ROOT_DIR)
ds_factory.prepare_data()
tvtsets = ds_factory.create_tvt_datasets(from_year=2010)
trainloader, validloader, testloader = (DataLoader(ds, batch_size=32, shuffle=True) for ds in tvtsets)
model = ConditionalAutoEncoder().double()
# import pdb; pdb.set_trace()

# train
MAX_EPOCH = 10
LEARNING_RATE = .001
L1_PENALTY = .001

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

size = len(trainloader.dataset)
for batch, (x0, x1, y) in enumerate(trainloader):
    pred = model(x0, x1)
    loss = loss_fn(pred, y)
    loss = loss + L1_PENALTY * sum(torch.norm(param.data) for nm, param in model.named_parameters())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch % 1000 == 0:
        loss, current = loss.item(), (batch+1) * len(x0)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# validate
size = len(validloader.dataset)
num_batches = len(validloader)
test_loss, correct = 0, 0

with torch.no_grad():
    for x0, x1, y in validloader:
        pred = model(x0, x1)
        test_loss += loss_fn(pred, y).item()

test_loss /= num_batches
# correct /= size
print(f"Test Error: Avg loss: {test_loss:>8f} \n")