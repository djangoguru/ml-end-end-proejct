'''
- Load config from config YAML file
- Initilize production database
 - Run the full training pipepline (preprocesssing, feature engineering. model training, postprocessing)
 - Save the training model to the models folder
'''

import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

sys.path.append(str(project_root / 'ml-app'/ 'src'))

os.chdir(project_root)



from common.utils import read_config
from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager


if __name__ == "__main__":
    config_path = project_root / 'config' / 'config.yaml'
    print(config_path)
    config = read_config(config_path)

    # Initialize production database with historical raw data
    data_manager = DataManager(config)
    data_manager.initiliaze_prod_database()

    # Initialse the pipeline runner

    pipeline_runner = PipelineRunner(config= config, data_manager= data_manager)

    #Run the full training pipeline
    pipeline_runner.run_training()
