'''Inference Pipeline
- Loads Configuration
- Initiialize the production database
- Loops through a given  umber of timestamaps 
- Save the predictions 
- Plot comparison of prediction and actual values
'''

import os
import sys
from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'ml-app'/ 'src'))


from common.utils import read_config, plot_predictions_vs_actual
from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager


if __name__ == "__main__":

    num_timestamps = 300

    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    data_manager = DataManager(config)
    current_timestamps = pd.to_datetime(config['pipeline_runner']['first_timestamp'])
    time_increment = pd.Timedelta(config['pipeline_runner']['time_increment'])

    data_manager.initiliaze_prod_database()

    pipeline_runner = PipelineRunner(config, data_manager)

    dataset_path = os.path.join(
       config['data_manager']['prod_data_folder'],
       config['data_manager']['real_time_data_prod_name']
    )

    df = data_manager.load_data(dataset_path)

    for i in range(num_timestamps):
        print(f"Processing timestamp: {i+1}/{num_timestamps}: {current_timestamps}")

        pipeline_runner.run_inference(current_timestamps)

        current_timestamps += time_increment


    print("Inference process completed.")

    print("Loading the data for plotting predictions vs actual values...")

    predictions_df = data_manager.load_prediction_data()
    actuals_df = data_manager.load_prod_data()
    plot_predictions_vs_actual(predictions_df, actuals_df)



