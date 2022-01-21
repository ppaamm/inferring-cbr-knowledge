import pandas as pd
from typing import List, Tuple


class DataLoader:
    
    def __init__(self, path: str, case: str):
        self.path = path
        self.case = case
        
        # Loading data
        df = pd.read_csv (self.path)
        # Filtering out invalid data
        self.df = df[df[self.case] != '—']


    def generate_CB(self, n_data_per_type: List[Tuple[int, int]]):
        sampled_df = [self.df[self.df.type == word_type[0]].sample(n=word_type[1]) for word_type in n_data_per_type]
        CB = pd.concat(sampled_df).filter(items = ['nominative', self.case]).values.tolist()
        X = [z[0] for z in CB]
        Y = [z[1] for z in CB]
        return CB, X, Y