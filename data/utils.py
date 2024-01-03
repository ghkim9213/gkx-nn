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

GKX_INDEX_COLS = ["permno", "DATE", "sic2"]
GKX_LABEL_COL = "ret"

class GKXDataset(Dataset):
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

        self._set_data() 

    @property
    def index_cols(self):
        return GKX_INDEX_COLS
    
    @property
    def characteristics(self):
        return self._chars.set_index(self.index_cols)
    
    @property
    def characteristics_cols(self):
        return self.characteristics.columns.tolist()
    
    @property
    def chunks(self):
        filepath = os.path.join(self.root_dir, GKX_CSV_FILENAME)
        if not os.path.exists(filepath):
            download_gkx_dataset(save_to=self.root_dir)
        return pd.read_csv(filepath, chunksize=self.chunksize)

    def _set_data(self):
        self._set_characteristics()
        self._set_stock_returns()
        self._set_portfolio_returns()
    
    def _set_characteristics(self):
        chunks = []
        print("loading characteristics...")
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
        self._df = pd.concat(chunks, axis=0, ignore_index=True)
        self._fillna_characteristics()
    
    def _fillna_characteristics(self):
        print("filling na...")
        for c in tqdm(self.characteristics_cols):
            self._df[c] = self._df.groupby("DATE")[c].transform(lambda x: x.fillna(x.mean()))
    
    def _set_stock_returns(self):
        print("loading stock returns")
        self.stock_returns = self._chars.groupby("permno").mom1m.shift(-2)
    
    def _set_portfolio_returns(self):
        for c in self.characteristics_cols:
            import pdb; pdb.set_trace()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        item = self.df.loc[idx]
        return torch.tensor(item[self.feature_cols].values), item[self.label_col]