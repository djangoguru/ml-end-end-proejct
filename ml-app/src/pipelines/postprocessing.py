'''
Handle the post-processing of model after training
- Save the trained model to disk
- Format and return predictions result during inference
'''


import pandas as pd
from common.utils import save_model


class PostProcessingPipeline:
    def __init__(self, config):
        self.config = config

    def run_train(self, model):
        # Save the trained model to disk
        model_path = self.config['pipeline_runner']['model_path']
        save_model(model, model_path)

    def run_inference(self,y_pred, current_timestamp):
        # Format predictions into a DataFrame
        timestamp = pd.to_datetime(current_timestamp) + pd.Timedelta(self.config['pipeline_runner']['time_increment'])
        df_pred = pd.DataFrame({
            'datetime': [timestamp],
            'prediction': [y_pred]
        })

        return df_pred