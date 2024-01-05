from typing import List, Union, Optional, Callable

from io import BytesIO, StringIO
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

from .utils import Spinner

GKX_DOWNLOAD_URL = "https://dachxiu.chicagobooth.edu/download/datashare.zip"
GKX_CSV_FILENAME = "datashare.csv"
GKX_INDEX_COLS = ["permno", "DATE", "sic2"]
GKX_YEAR_RANGE = [str(y) for y in list(range(1957, 2022, 1))]

GKX_DEFAULT_STOCK_RETURN_COL = "r"

normalize = lambda x: (x-x.mean()) / x.std()
minmax = lambda x: (x-x.min()) / (x.max() - x.min())

class GKXDatasetFactory:
    def __init__(self, root_dir: str):
        assert os.path.exists(root_dir)
        self.root_dir = root_dir

    @property
    def root_file(self):
        return os.path.join(self.root_dir, GKX_CSV_FILENAME)

    @property
    def available_years(self):
        return GKX_YEAR_RANGE

    def download_data(self):
        if os.path.exists(self.root_file):
            warnings.warn("It looks there already exists the root file. Download not implemented.")
            return
        res = requests.get(GKX_DOWNLOAD_URL, verify=False)
        zipfile.ZipFile(BytesIO(res.content)).extract(GKX_CSV_FILENAME, path=self.root_dir)
            
    def split_by_year(
            self,
            split_ratio: List[float],
            from_year: Optional[str]=None,
            to_year: Optional[str]=None,
            with_portfolio_returns: bool=True,
            **kwargs
        ):
        assert len(split_ratio) > 1
        assert sum(split_ratio) == 1.
        if from_year is None:
            from_year = self.available_years[0]
        assert from_year in self.available_years
        if to_year is None:
            to_year = self.available_years[-1]
        assert to_year in self.available_years

        # split years
        from_loc = self.available_years.index(from_year)
        to_loc = self.available_years.index(to_year) + 1
        years = self.available_years[from_loc:to_loc]
        min_year = min(years)
        splitted_years = []
        total_n = len(years)
        for r in split_ratio:
            n = round(total_n*r)
            s, years = years[:n], years[n:]
            splitted_years.append(s)

        # split csv by years
        csv_handle = open(self.root_file, "r")
        buffer = StringIO(newline=None)

        reader = csv.reader(csv_handle)
        writer = csv.writer(buffer)
        header = next(reader)        

        print(f"seeking min year of {self.root_file}...")
        with Spinner():
            curr_year = "1900"
            while curr_year < min_year:
                row = next(reader)
                curr_year = row[1][:4]

        writer.writerows([header, row])
        dfs = []
        end_of_file = False
        for years in splitted_years:
            print(f"loading data of {years}...")
            with Spinner():
                max_year = max(years)
                while curr_year <= max_year:
                    try:
                        row = next(reader)
                    except StopIteration:
                        end_of_file = True
                        break
                    writer.writerow(row)
                    curr_year = row[1][:4]
                if end_of_file:
                    break
                # writer.writerow(row)

                # overlap two more month to preserve return
                overlapped_buffer = StringIO(newline=None)
                overlapped_writer = csv.writer(overlapped_buffer)
                overlapped_writer.writerows([header, row])

                max_date = str(int(max_year)+1) + "0300"
                curr_date = row[1]
                while True:
                    row = next(reader)
                    curr_date = row[1]
                    if curr_date > max_date:
                        break
                    writer.writerow(row)
                    overlapped_writer.writerow(row)
                buffer.seek(0)
                dfs.append(pd.read_csv(buffer, low_memory=False))
                buffer.seek(0)
                buffer.truncate(0)
                buffer.write(overlapped_buffer.getvalue())
                overlapped_buffer.close()
        if end_of_file:
            buffer.seek(0)
            dfs.append(pd.read_csv(buffer, low_memory=False))
        csv_handle.close()
        buffer.close()
        print("dataframes were successfully loaded!")

        cls = GKXDatasetWithPortfolioReturns if with_portfolio_returns else GKXDataset
        return tuple(cls(df=df, **kwargs) for df in dfs)


class GKXDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            scaling_func: Optional[Callable]=None,
        ):
        super().__init__()
        self._df = df
        self.scaling_func = scaling_func

        self._fillna_characteristics()
        self._scale_characteristics()
        self._set_stock_returns()
        
    @property
    def index_cols(self) -> List[str]:
        return GKX_INDEX_COLS
    
    @property
    def stock_return_col(self) -> str:
        return GKX_DEFAULT_STOCK_RETURN_COL
        
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

    def _scale_characteristics(self):
        print("scaling characteristics...")
        for c in tqdm(self.char_cols):
            self._df[c] = self._df.groupby("DATE")[c].transform(self.scaling_func)

    def _set_stock_returns(self):
        print("setting stock returns...")
        self._df[self.stock_return_col] = self._df.groupby("permno").mom1m.shift(-2)
        self._df = self._df.dropna(subset=[self.stock_return_col]).reset_index(drop=True)

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx):
        item = self._df.loc[idx]
        return torch.tensor(item[self.char_cols].values), item[self.stock_return_col]


class GKXDatasetWithPortfolioReturns(GKXDataset):
    def __init__(self, df, scaling_func=None, quantiles: int=5):
        super().__init__(df=df, scaling_func=scaling_func)
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