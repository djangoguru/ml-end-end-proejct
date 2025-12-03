'''
It will have a class called PipelineRunner

'''
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

sys.path.append(str(project_root / 'ml-app'/ 'src'))

from common.data_manager import DataManager
from pipelines.feature_engineeing import FeatureEngineeringPipeline
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.postprocessing import PostProcessingPipeline
import pandas



class PipelineRunner:
    def __init__(self, config, data_manager):
        self.config = config
        self.data_manager = data_manager

        #Initialize the individual pipeline components here
        self.preprocessing_pipeline = PreprocessingPipeline(config = config)
        self.feature_eng_pipeline = FeatureEngineeringPipeline(config = config)
        self.training_pipeline = TrainingPipeline(config = config)
        self.inference_pipeline = InferencePipeline(config = config)
        self.postprocessing_pipeline = PostProcessingPipeline(config = config)

        #Load the real-time data

        self.real_time_data = self.data_manager.load_data(
            os.path.join(
                config['data_manager']['prod_data_folder'],
                config['data_manager']['real_time_data_prod_name']
            )
        )

        self.prod_data_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['prod_database_name']
        )

        self.current_database_data = self.data_manager.load_data(self.prod_data_path)


    def run_training(self):

        df = self.data_manager.load_data(self.prod_data_path)
        df = self.preprocessing_pipeline.run(df = df)
        df = self.feature_eng_pipeline.run(df = df)
        model = self.training_pipeline.run(df = df)
        self.postprocessing_pipeline.run_train(model = model)
        return


    def run_inference(self, current_timestamp):


        #Step 1 = Retieve real-time data for the current timestamp
        current_real_time_data = self.data_manager.get_time_stamp_data(
            data = self.real_time_data,
            timestamp = current_timestamp
        )

        #Step 2: Append new data to prod database
        self.current_database_data = self.data_manager.append_data(
            current_data = self.current_database_data,
            new_data = current_real_time_data
        )


        #Step 3:  Get the last N rows as the latest batch

        df = self.data_manager.get_n_last_data_points(
            data = self.current_database_data,
            n = self.config['pipeline_runner']['batch_size'])
        

        #Step 4: Run preprocessing and feature engineering

        df = self.preprocessing_pipeline.run(df = df)
        df = self.feature_eng_pipeline.run(df = df)

        #Step 5: Run inference
        y_pred = self.inference_pipeline.run(x = df)

        #Step 6: Postprocess and save predictions
        df_pred = self.postprocessing_pipeline.run_inference(
            y_pred = y_pred,
            current_timestamp = current_timestamp
        )

        #Step 7: save the predictions and update database to access in the UI application

        self.data_manager.save_predictions(df_pred, current_timestamp)
        self.data_manager.save_data(data= self.current_database_data, path = self.prod_data_path)
        return
