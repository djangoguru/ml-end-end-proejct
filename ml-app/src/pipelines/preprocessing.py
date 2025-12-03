'''
Handle the preprocessing steps:
- Column Renaming
-  Column Dropping

'''

import pandas as pd


class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config['preprocessing']

    @staticmethod
    def rename_columns(df, column_mapping):
        return df.rename(columns=column_mapping)
    
    @staticmethod
    def drop_columns(df, columns):
        df.drop(columns=columns, inplace=True)
        return df

    def run(self, df):
        # Example preprocessing steps

        df.reset_index(drop=True, inplace=True)
        df = self.rename_columns(df, self.config['column_mapping'])
        df = self.drop_columns(df, self.config['drop_columns'])
        return df
