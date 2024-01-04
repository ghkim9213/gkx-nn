from typing import List, Union

from io import BytesIO
from itertools import accumulate
import csv
import os
import warnings
import zipfile

from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
import torch
import requests
import pandas as pd

GKX_DOWNLOAD_URL = "https://dachxiu.chicagobooth.edu/download/datashare.zip"
GKX_CSV_FILENAME = "datashare.csv"


GKX_INDEX_COLS = ["permno", "DATE", "sic2"]
GKX_STOCK_RETURN_COL = "r"
GKX_DEFAULT_SOURCE_NAME = "datashare.csv"
GKX_DEFAULT_CHUNKED_DIR = "chunked_by_year"
class GKXDatasetFactory:
    def __init__(self, root_dir: str):
        assert os.path.exists(root_dir)
        self.root_dir = root_dir

    @property
    def root_file(self):
        return os.path.join(self.root_dir, GKX_DEFAULT_SOURCE_NAME)

    @property
    def chunked_dir(self):
        return os.path.join(self.root_dir, GKX_DEFAULT_CHUNKED_DIR)

    def prepare_data(self):
        self.download()
        self.chunk_by_year()

    def download(self):
        if os.path.exists(self.root_file):
            warnings.warn("It looks there already exists the root file. Download not implemented.")
            return
        res = requests.get(GKX_DOWNLOAD_URL, verify=False)
        zipfile.ZipFile(BytesIO(res.content)).extract(GKX_CSV_FILENAME, path=self.root_dir)
    
    def chunk_by_year(self):
        assert os.path.exists(self.root_file)
        if os.path.exists(os.path.join(self.chunked_dir)):
            warnings.warn("It looks there already exists the chunked files. Chunking not implemented.")
            return
        os.mkdir(os.path.join(self.chunked_dir))
        with open(self.root_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            prev_year = "9999"
            data = [header]
            print("chunking by year...")
            for row in tqdm(reader):
                if (not row[1].startswith(prev_year)) and (len(data) > 1):
                    self._write_csv( # write data of prev year
                        os.path.join(self.chunked_dir, f"{prev_year}.csv"),
                        data
                    ) 
                    data = [header] # clear container
                data.append(row)
                prev_year = row[1][:4]
            self._write_csv(
                os.path.join(self.chunked_dir, f"{prev_year}.csv"),
                data
            ) 
    
    def _write_csv(self, filepath, data):
        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def create_dataset(
            self,
            years: Union[int, List[int]],
            with_portfolio_returns: bool=True
        ):
        cls = GKXDatasetWithPortfolioReturns if with_portfolio_returns else GKXDataset
        return cls(
            root_dir=self.chunked_dir,
            years=years,
        )

    @property
    def available_years(self):
        return sorted([int(fnm.split(".")[0][-4:]) for fnm in os.listdir(self.chunked_dir)])
    
    def create_tvt_datasets(
            self,
            from_year: int,
            split_ratio: List[float]=[.3, .2, .5],
            with_portfolio_returns: bool=True
        ):
        assert sum(split_ratio) == 1.
        from_loc = self.available_years.index(from_year)
        years = self.available_years[from_loc:]
        train_loc, valid_loc, _ = accumulate(int(len(years) * r) for r in split_ratio)
        tvt_years = (
            years[:train_loc],
            years[train_loc:valid_loc],
            years[valid_loc:],
        )
        return (self.create_dataset(ys, with_portfolio_returns) for ys in tvt_years)


class GKXDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            years: Union[int, List[int]],
            minmax_scale: bool = True,
        ):
        super().__init__()
        self.root_dir = root_dir
        if isinstance(years, int):
            years = [years]
        self.years = sorted(years)

        # consider one more years to fully get shifted returns
        self.next_year = max(years) + 1

        self._df = self._load_dataframe()
        self._fillna_characteristics()
        if minmax_scale:
            self._minmax_scale_characteristics()
        self._set_stock_returns()
    
    def _load_dataframe(self):
        print("loading data...")
        dfs = []
        for path in self.filepaths:
            if not os.path.exists(path):
                continue
            dfs.append(pd.read_csv(path))
        return pd.concat(dfs, axis=0, ignore_index=True)
    
    @property
    def filepaths(self):
        return [os.path.join(self.root_dir, f"{y}.csv") for y in self.years + [self.next_year]]

    @property
    def index_cols(self) -> List[str]:
        return GKX_INDEX_COLS
    
    @property
    def stock_return_col(self) -> str:
        return GKX_STOCK_RETURN_COL
    
    @property
    def char_cols(self) -> List[str]:
        non_char_cols = self.index_cols + [self.stock_return_col]
        return [c for c in self._df.columns if c not in non_char_cols]
    
    @property
    def characteristics(self):
        return self._df.set_index(self.index_cols)[self.char_cols]
    
    @property
    def stock_returns(self):
        return self._df.set_index(self.index_cols)[self.stock_return_col]
    
    def _fillna_characteristics(self):
        print("filling na...")
        for c in tqdm(self.char_cols):
            self._df[c] = self._df.groupby("DATE")[c].transform(lambda x: x.fillna(x.mean()))

    def _minmax_scale_characteristics(self):
        print("minmax scaling...")
        # for c in tqdm(self.char_cols):
        #     self._df[c] = self._df.groupby("DATE")[c].transform(lambda x: (x - x.min())/ (x.max() - x.min()))
        for c in tqdm(self.char_cols):
            self._df[c] = self._df.groupby("DATE")[c].transform(lambda x: (x - x.mean())/ x.std())

    def _set_stock_returns(self):
        print("setting stock returns...")
        self._df[self.stock_return_col] = self._df.groupby("permno").mom1m.shift(-2)
        isin_years = self._df["DATE"].astype(str).str[:4].astype(int).isin(self.years)
        self._df = self._df.loc[isin_years].dropna(subset=[self.stock_return_col]).reset_index(drop=True)

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx):
        item = self._df.loc[idx]
        return torch.tensor(item[self.char_cols].values), item[self.stock_return_col]


class GKXDatasetWithPortfolioReturns(GKXDataset):
    def __init__(self, root_dir, years, quantiles=5):
        super().__init__(root_dir, years)
        self.quantiles = quantiles
        self._set_portfolio_returns()

    def _set_portfolio_returns(self):
        r_longshorts = []
        print("caculating portfolio returns...")
        for c in tqdm(self.char_cols):
            group = self._df.groupby("DATE")[c].transform(
                lambda x: pd.qcut(x, self.quantiles, labels=False, duplicates="drop")
            )
            r_pf = self._df.groupby(["DATE", group])[self.stock_return_col].apply(lambda x: x.mean()).unstack(c)
            r_pf = r_pf.T.ffill().T # fill na from dropping duplicates
            r_longshort = r_pf.iloc[:, -1] - r_pf.iloc[:,0]
            r_longshorts.append(r_longshort.rename(c))
        self.portfolio_returns = pd.concat(r_longshorts, axis=1)
    
    def __getitem__(self, idx):
        item = self._df.loc[idx]
        chars = torch.tensor(item[self.char_cols].values)
        factors = torch.tensor(self.portfolio_returns.loc[item["DATE"]].values)
        stock_return = item[self.stock_return_col]
        return chars, factors, stock_return