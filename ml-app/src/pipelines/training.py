'''
- This pipeline will be used for training and optimizing machine learning models.
- We will use specifically CatBoost, using configuration-driven parametrs and Optina for hyperparameter optimization
'''
import optuna
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import optuna

class TrainingPipeline:

    def __init__(self, config):
        self.config = config['training']
        self.optuna_confif = self.config.get('optuna', {})
        self.search_space = self.config['optuna']['search_space']

    @staticmethod
    def make_target(df, target_params):
        shift_period = target_params['shift_period']
        df[target_params['new_target_name']] = df[target_params['target_column']].shift(-shift_period).ffill()
        return df

    def prepare_dataset(self, df):
        df = self.make_target(df, target_params=self.config['target_params'])
        feats = [col for col in df.columns if col != self.config['target_params']['new_target_name']]
        x, y = df[feats], df[self.config['target_params']['new_target_name']]

        train_size = int(self.config['train_fraction'] * len(df))
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return x_train, x_test, y_train, y_test
    
    def tune_hyperparams(self, x_train, y_train, x_test, y_test):

        np.random.seed(42)

        def objective(trial):
            ss = self.search_space
            params = {
                'learning_rate': trial.suggest_float(
                    'learning_rate', ss['learning_rate']['low'], ss['learning_rate']['high'], 
                    log=ss['learning_rate']['log']
                    ),
                'depth': trial.suggest_int('depth', ss['depth']['low'], ss['depth']['high']),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 
                                                   ss['l2_leaf_reg']['low'], ss['l2_leaf_reg']['high'],),
                'iterations': self.config['iterations'],
                'loss_function': self.config['loss_function'],
                'verbose': self.config['verbose']}
            


            #Manual time-based validation splt

            train_idx = int(self.config['train_fraction'] * len(x_train))
            x_tr, x_val = x_train.iloc[:train_idx], x_train[train_idx:]
            y_tr, y_val = y_train.iloc[:train_idx], y_train[train_idx:]

            
            model = CatBoostRegressor(**params, random_state =42, allow_writing_files=False)
            model.fit(x_tr, y_tr, eval_set=(x_val, y_val), early_stopping_rounds= self.config.get('early_stopping_rounds', 100), 
                      use_best_model=True, verbose=False)
            


            
            preds = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            trial.set_user_attr('best_iteration', model.get_best_iteration())
            return rmse
        
        #Run the OPtuna study

        n_trials = self.optuna_confif['n_trials']
        study = optuna.create_study(direction='minimize', sampler = optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        #Train the final model with the best hyperparameters on the full training data

        best_params = study.best_params.copy()
        best_params.update({
            'iterations': study.best_trial.user_attrs['best_iteration'],
            'loss_function': self.config['loss_function'],
            'verbose': False
        })

        final_model = CatBoostRegressor(**best_params, random_state=42, allow_writing_files=False)
        final_model.fit(x_train, y_train, verbose = False)

        return final_model, study
    



    def run(self, df):
        x_train, x_test, y_train, y_test = self.prepare_dataset(df)
        model, _ = self.tune_hyperparams(x_train, y_train, x_test, y_test)
        return model