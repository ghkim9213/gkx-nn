from typing import Optional

from io import BytesIO
import os
import warnings
import zipfile

from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import requests
import pandas as pd

GKX_DOWNLOAD_URL = "https://dachxiu.chicagobooth.edu/download/datashare.zip"
GKX_CSV_FILENAME = "datashare.csv"

def download_gkx_dataset(save_to="."):
    res = requests.get(GKX_DOWNLOAD_URL, verify=False)
    zipfile.ZipFile(BytesIO(res.content)).extract(GKX_CSV_FILENAME, path=save_to)


class GKXDataset(Dataset):
    INDEX_COLS = ["permno", "DATE", "sic2"]
    FEATURE_COLS = None
    LABEL_COL = "ret"

    def __init__(
            self,
            root_dir: str,
            from_date: Optional[str]=None,
            to_date: Optional[str]=None,
            chunksize: int=1000,
        ):
        super().__init__()
        self.root_dir = root_dir
        if from_date is None and to_date is None:
            warnings.warn("'from_date' and 'to_date' were not set.")
        self.from_date = from_date
        self.to_date = to_date
        self.chunksize = chunksize
        self._set_dataframe()

    @property
    def chunks(self):
        filepath = os.path.join(self.root_dir, GKX_CSV_FILENAME)
        if not os.path.exists(filepath):
            download_gkx_dataset(save_to=self.root_dir)
        return pd.read_csv(filepath, chunksize=self.chunksize)
    
    def _set_dataframe(self):
        chunks = []
        print("loading dataframe...")
        for chunk in tqdm(self.chunks):
            fltr = None
            if self.from_date is not None:
                fltr = chunk["DATE"] >= int(self.from_date)
            if self.to_date is not None:
                fltr =  fltr & (chunk["DATE"] <= int(self.to_date))
            if fltr is not None:
                chunk = chunk.loc[fltr]
            if len(chunk) > 0:
                chunks.append(chunk.loc[fltr])
        df = pd.concat(chunks, axis=0, ignore_index=True)
        df = df.groupby("DATE").transform(lambda x: x.fillna(x.mean()))
        df[self.LABEL_COL] = df.groupby("permno").mom1m.shift(-2)        
        self.FEATURE_COLS = [c for c in df.columns if c not in [self.LABEL_COL]+self.INDEX_COLS]        
        self.df = df
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        item = self.df.loc[idx]
        return torch.tensor(item[self.FEATURE_COLS].values), item[self.LABEL_COL]