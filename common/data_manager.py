'''
- Handle all the data related I/O operation
- Trasformations used accross ML pipeline


Responsibilties: 
- Initilize the production database
- Load and save the parquet files
- Appending the new data to exsisting dataset
- Slicing and filtering
- Save the predictoins incrementally.

'''
import os
import sys
from pathlib import Path
import pandas as pd


class DataManager:
    def __init__(self, config):
        self.config = config

    def initiliaze_prod_database(self):
        raw_data_path = os.path.join(
            self.config['data_manager']['raw_data_folder'], 
            self.config['data_manager']['raw_database_name']
            )
            
        prod_data_path = os.path.join(
            self.config['data_manager']['prod_data_folder'], 
            self.config['data_manager']['prod_database_name']
            )
        
        df = pd.read_parquet(raw_data_path)

        df.to_parquet(prod_data_path, index=False)

        prediction_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['real_time_prediction_data_name']
        )


        if os.path.exists(prediction_path):
            os.remove(prediction_path)

    @staticmethod
    def load_data(path):
        return pd.read_parquet(path)
    

    def load_prediction_data(self):
        prediction_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['real_time_prediction_data_name']
        )

        df = self.load_data(prediction_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def load_prod_data(self):
        prod_data_path = os.path.join(
            self.config['data_manager']['prod_data_folder'], 
            self.config['data_manager']['prod_database_name']
            )
        
        df = self.load_data(prod_data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    @staticmethod
    def get_time_stamp_data(data, timestamp):
        return data.loc[pd.to_datetime(data['datetime']) == pd.to_datetime(timestamp)]
    
    @staticmethod
    def append_data(current_data, new_data):

        df = pd.concat([current_data, new_data], axis = 0)
        df.reset_index(drop = True, inplace = True)
        return df
    
    @staticmethod
    def get_n_last_data_points(data, n):
        return data.iloc[-n:]
    
    def save_predictions(self, pred_df, current_timestamp):

        prediction_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['real_time_prediction_data_name']
        )

        if os.path.exists(prediction_path):
            if current_timestamp == pd.to_datetime(self.config['pipeline_runner']['first_timestamp']):
                combined_df = pred_df

            else:
                exsisting_pred_df = pd.read_parquet(prediction_path)
                combined_df = pd.concat([exsisting_pred_df, pred_df], axis = 0, ignore_index = True)

        #File does not exsist
        else:
            combined_df = pred_df

        #Save final DataFrame

        combined_df.to_parquet(prediction_path, index = False)

    @staticmethod
    def save_data(data, path):
        data.to_parquet(path, index = False)