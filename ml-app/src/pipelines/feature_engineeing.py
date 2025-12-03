'''
Here we are going create and engineer the features coming from preprocessing step
- Create the lag features for time series data
'''


import pandas as pd


class FeatureEngineeringPipeline:

    def __init__(self, config):
        self.config = config['feature_engineering']

    @staticmethod
    def add_lag_feats(df, params):
        for feat, lags in params.items():
            for lag in lags:
                df[f'{feat}_lag_{lag}'] = df[feat].shift(lag).bfill()
        return df
    

    def run(self, df):
        df = self.add_lag_feats(df,self.config['lag_params'])
        return df