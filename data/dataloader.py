import pandas as pd
from typing import List, Tuple


class DataLoader:
    
    def __init__(self, path: str, source_case: str, target_case: str):
        self.path = path
        self.source_case = source_case
        self.target_case = target_case
        
        # Loading data
        self.df = pd.read_csv (self.path)
        # Filtering out invalid data
        self.df = self.df[self.df[self.source_case] != '—'] 
        self.df = self.df[self.df[self.target_case] != '—']


    def generate_CB(self, n_data_per_type: List[Tuple[int, int]]):
        sampled_df = [self.df[self.df.type == word_type[0]].sample(n=word_type[1]) for word_type in n_data_per_type]
        CB = pd.concat(sampled_df).filter(items = [self.source_case, self.target_case]).values.tolist()
        X = [z[0] for z in CB]
        Y = [z[1] for z in CB]
        return CB, X, Y