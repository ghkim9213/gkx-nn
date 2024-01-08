from typing import Optional, Callable, Sequence
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

default_input_extractor = lambda loaded: loaded[0]
default_label_extractor = lambda loaded: loaded[1]

def _format_input(input):
    if isinstance(input, torch.Tensor):
        return tuple([input])
    elif isinstance(input, Sequence):
        return tuple(inp for inp in input)

def train_epoch(
        model: torch.nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
    ):
    model = model.to(device)
    model.train()
    if input_extractor is None:
        input_extractor = default_input_extractor
    if label_extractor  is None:
        label_extractor = default_label_extractor
    train_loss = 0.
    for batch_count, loaded in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = tuple(inp.to(device) for inp in _format_input(input_extractor(loaded)))
        y = label_extractor(loaded).to(device)
        optimizer.zero_grad()
        pred = model(*x)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= batch_count
    return train_loss

def validate_epoch(
        model: torch.nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        loss_fn: Callable,
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
    ):
    model = model.to(device)
    model.eval()
    if input_extractor is None:
        input_extractor = default_input_extractor
    if label_extractor  is None:
        label_extractor = default_label_extractor
    valid_loss = 0.
    with torch.no_grad():
        for batch_count, loaded in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = tuple(inp.to(device) for inp in _format_input(input_extractor(loaded)))
            y = label_extractor(loaded).to(device)
            pred = model(*x)
            valid_loss += loss_fn(pred, y).item()
    valid_loss /= batch_count
    return valid_loss
