import sys
import os
from flask import Flask, jsonify
from pathlib import Path
import pandas as pd


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root ,'src'))
sys.path.append(os.path.join(project_root, 'ml-app', 'src'))
os.chdir(project_root)

from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager

app = Flask(__name__)

@app.route('/run-inference', methods=['POST'])
def run_inference():
    try:
        dataseet_path = os.path.join(
            config['data_manager']['prod_data_folder'],
            config['data_manager']['prod_database_name']
        )

        df_prod = data_manager.load_data(dataseet_path)
        df_prod['datetime'] = pd.to_datetime(df_prod['datetime'])

        latest_timestamp = df_prod['datetime'].max()
        time_increment = pd.Timedelta(config['pipeline_runner']['time_increment'])
        current_timestamp = latest_timestamp + time_increment

        pipeline_runner.run_inference(current_timestamp)
        return jsonify({'status': 'success', "timestamp": str(current_timestamp)}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})

if __name__ == "__main__":
    
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)
    print(config_path)

    data_manager = DataManager(config)
    data_manager.initiliaze_prod_database()

    pipeline_runner = PipelineRunner(config, data_manager)

    app.run(host = "0.0.0.0", port = 5001)

