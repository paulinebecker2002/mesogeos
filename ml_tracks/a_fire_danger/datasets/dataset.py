from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import random
import warnings


class FireDataset(Dataset):
    def __init__(self, dataset_root: Path = None, problem_class: str = 'classification', train_val_test: str = 'train',
                 dynamic_features: list = None, static_features: list = None, nan_fill: float = 0,
                 neg_pos_ratio: int = 2, lag: int = 30, seed: int = 12345, last_n_timesteps: int = 30,
                 train_year: list = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'],
                 val_year: list = ['2020'], test_year: list = ['2021', '2022'],
                 pos_source: str = 'positives.csv',
                 neg_source: str = 'negatives.csv',
                 coastal_only: bool = False, inland_only: bool = False,
                 n_train_pos: Optional[int] = None,
                 n_val_pos:   Optional[int] = None,
                 n_test_pos:  Optional[int] = None,):
        """
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2019].
                'val' gets samples from 2020.
                'test' get samples from 2021-2022.
        @param dynamic_features: selects the dynamic features to return
        @param static_features: selects the static features to return
        @param categorical_features: selects the categorical features
        @param nan_fill: Fills nan with the value specified here
        """
        dataset_root = Path(dataset_root)

        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.problem_class = problem_class
        self.nan_fill = nan_fill
        self.lag = lag
        self.seed = seed
        self.last_n_timesteps = last_n_timesteps
        self.train_year = train_year
        self.val_year = val_year
        self.test_year = test_year
        self.pos_source = pos_source
        self.neg_source = neg_source
        self.coastal_only = coastal_only
        self.inland_only = inland_only
        self.n_train_pos = n_train_pos
        self.n_val_pos = n_val_pos
        self.n_test_pos = n_test_pos

        random.seed(self.seed)
        np.random.seed(self.seed)

        assert problem_class in ['classification', 'segmentation']
        assert self.pos_source in ['positives.csv', 'positives_binary_coastal.csv']
        assert self.neg_source in ['negatives.csv', 'negatives_binary_coastal.csv']
        assert not (coastal_only and inland_only), "You cannot set both `coastal=True` and `inland=True` at the same time."


        pos_csv = dataset_root / self.pos_source
        neg_csv = dataset_root / self.neg_source

        print(f"Loading positives from: {pos_csv}")
        print(f"Loading negatives from: {neg_csv}")

        self.positives = pd.read_csv(pos_csv)
        self.positives['label'] = 1
        self.negatives = pd.read_csv(neg_csv)
        self.negatives['label'] = 0

        # Filter by coastal or inland selection
        if coastal_only:
            print("Filtering: keeping only coastal samples (coastal == 1)")
            self.positives = self.positives[self.positives['coastal'] == 1]
            self.negatives = self.negatives[self.negatives['coastal'] == 1]
        elif inland_only:
            print("Filtering: keeping only inland samples (coastal == 0)")
            self.positives = self.positives[self.positives['coastal'] == 0]
            self.negatives = self.negatives[self.negatives['coastal'] == 0]


        for df in (self.positives, self.negatives):
            if 'coastal' not in df.columns:
                warnings.warn("'coastal' column not found; defaulting to 0.")
                df['coastal'] = 0
            df['coastal'] = df['coastal'].astype(int).clip(0, 1)

        def get_last_year(group):
            last_date = group['time'].iloc[-1]
            return last_date[:4]

        self.positives['YEAR'] = self.positives.groupby(self.positives.index // 30).apply(get_last_year).values.repeat(30)
        self.negatives['YEAR'] = self.negatives.groupby(self.negatives.index // 30).apply(get_last_year).values.repeat(30)

        self.train_positives = self.positives[self.positives['YEAR'].isin(self.train_year)]
        self.val_positives = self.positives[self.positives['YEAR'].isin(self.val_year)]
        self.test_positives = self.positives[self.positives['YEAR'].isin(self.test_year)]
        print(f"Training in years: {self.train_year}")
        print(f"Validation in years: {self.val_year}")
        print(f"Testing in years: {self.test_year}")

        def limit_positives(pos_df: pd.DataFrame, n_limit: Optional[int]) -> pd.DataFrame:
            if n_limit is None:
                return pos_df
            pos_ids = pos_df['sample'].unique()
            if n_limit > len(pos_ids):
                raise ValueError(f"Proposed amount of positives samples: {n_limit}, but only {len(pos_ids)} available.")
            chosen = np.random.choice(pos_ids, size=n_limit, replace=False)
            return pos_df[pos_df['sample'].isin(chosen)]

        if self.n_train_pos is not None:
            self.train_positives = limit_positives(self.train_positives, self.n_train_pos)
        if self.n_val_pos is not None:
            self.val_positives   = limit_positives(self.val_positives,   self.n_val_pos)
        if self.n_test_pos is not None:
            self.test_positives  = limit_positives(self.test_positives,  self.n_test_pos)

        bas_median = self.train_positives['burned_area_has'].median()

        def random_(negatives, positives, neg_pos_ratio):
            n_pos_samples = len(positives) // 30
            n_neg_required = n_pos_samples * neg_pos_ratio
            unique_samples = negatives['sample'].unique()
            replace = len(unique_samples) < n_neg_required
            ids_selected = np.random.choice(unique_samples, n_neg_required, replace=replace)
            return negatives[negatives['sample'].isin(ids_selected)]

        self.train_negatives = random_(self.negatives[self.negatives['YEAR'].isin(self.train_year)],
                                       self.train_positives, neg_pos_ratio)
        self.val_negatives = random_(self.negatives[self.negatives['YEAR'].isin(self.val_year)], self.val_positives,
                                     neg_pos_ratio)
        self.test_negatives = random_(self.negatives[self.negatives['YEAR'].isin(self.test_year)], self.test_positives,
                                      neg_pos_ratio)

        if train_val_test == 'train':
            print(f'Train Positives: {len(self.train_positives) / 30} / Train Negatives: {len(self.train_negatives) / 30}')
            self.all = pd.concat([self.train_positives, self.train_negatives]).reset_index()
            print("Training Dataset length", len(self.all) / 30)
        elif train_val_test == 'val':
            print(f'Validation Positives: {len(self.val_positives) / 30} / Validation Negatives: {len(self.val_negatives) / 30}')
            self.all = pd.concat([self.val_positives, self.val_negatives]).reset_index()
            print("Validation Dataset length", len(self.all) / 30)
        elif train_val_test == 'test':
            print(f'Test Positives: {len(self.test_positives) / 30} / Test Negatives: {len(self.test_negatives) / 30}')
            self.all = pd.concat([self.test_positives, self.test_negatives]).reset_index()
            print("Test Dataset length", len(self.all) / 30)


        self.labels = self.all.label.tolist()
        self.samples = self.all['sample'].tolist()
        self.dynamic = self.all[self.dynamic_features]
        self.static = self.all[self.static_features]

        # Normalize the dynamic and static features
        self.dynamic = (self.dynamic - self.dynamic.mean()) / self.dynamic.std()
        self.static = (self.static - self.static.mean()) / self.static.std()

        self.burned_areas_size = self.all['burned_area_has'].replace(0, 30)

        self.coastal = self.all['coastal'].astype(int)


    def __len__(self):
        return int(len(self.all) / 30)


    def __getitem__(self, idx):
        full_dynamic = self.dynamic.iloc[idx * 30:(idx + 1) * 30].values[-self.lag:, :]
        dynamic = full_dynamic[-self.last_n_timesteps:, :]

        static = self.static.iloc[idx * 30:(idx + 1) * 30].values[0, :]
        burned_areas_size = np.log(self.burned_areas_size.iloc[idx * 30:(idx + 1) * 30].values[0])
        labels = self.labels[idx * 30]
        sample_id = self.samples[idx * 30]
        x = self.all.iloc[idx * 30]['x']
        y = self.all.iloc[idx * 30]['y']
        coastal = int(self.coastal.iloc[idx * 30])


        # Fill NaN values with the mean of the column
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat_mean = np.nanmean(dynamic, axis=0)
            # Find indices that you need to replace
            inds = np.where(np.isnan(dynamic))
            # Place column means in the indices. Align the arrays using take
            dynamic[inds] = np.take(feat_mean, inds[1])

        dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
        static = np.nan_to_num(static, nan=self.nan_fill)

        return dynamic, static, burned_areas_size, labels, x, y, sample_id, coastal