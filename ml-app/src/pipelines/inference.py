'''
- Load the trained model
- Prepare input data for inference
- Generate predictions
- Post process the predictions
'''


from common.utils import load_model
import pandas as pd 

class InferencePipeline:
    def __init__(self, config):
        self.config= config


    def run(self, x):

        model = load_model(base_path=self.config['pipeline_runner']['model_path'])
        y_pred = model.predict(x)

        y_pred = y_pred[-1]
        return y_pred