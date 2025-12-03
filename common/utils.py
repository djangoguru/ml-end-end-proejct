'''
- Read the YAML file
- Load the model
- Save the model to the models folder

'''

import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
import matplotlib.dates as mdates
import pickle
import plotly.graph_objects as go
import logging 



def read_config(path) -> dict:
    path = Path(path)
    print(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: Please specify the config file {path}")
    
    with path.open('r') as f:
        
        return yaml.safe_load(f)
    
def plot_predictions_vs_actual(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame, save_path: str = 'inference_results.png'):
    merged = pd.merge(predictions_df, actuals_df, on='datetime', how = 'inner')
    if merged.empty:
        print("No overlapping datetime values between predictions and actuals for plotting.")
        return
    
    metrics = calculate_prediction_metrics(predictions_df, actuals_df)
    rmse_text = f" (RMSE: {metrics['rmse']:.2f}" if metrics['rmse'] is not None else "RMSE: N/A"

    plt.figure(figsize=(14, 6))
    plt.plot(merged['datetime'], merged['cnt'], label = 'Actual Bike Count', color = 'blue', marker = 'o', markersize = 4)
    plt.plot(merged['datetime'], merged['prediction'], label='Predicted Bike Count', color='orange',marker = 'o', markersize = 4)
    plt.xlabel('Time')
    plt.ylabel('Bike Count')
    plt.title(f'Bike Count->Predictions vs Actual Values {rmse_text} ')
    plt.legend()
    plt.grid(True)


    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks = 10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=30)
    plt.subplots_adjust(bottom=0.22)

    plt.tight_layout()
    plt.show()

def load_model(base_path):
    path = Path(base_path)
    cbm_path = path.with_suffix('.cbm')
    pkl_path = path.with_suffix('.pkl')

    if cbm_path.exists():
        model = CatBoostRegressor()
        model.load_model(str(cbm_path))
        return model
    
    elif pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    else:
        raise FileNotFoundError(f"No model file found at {base_path} with .cbm or .pkl extension.")
        data_prepared = self.prepare_data(x)


def save_model(model, base_path):
    path = Path(base_path)
    if isinstance(model, CatBoostRegressor):
        model.save_model(path.with_suffix('.cbm'))
        print(f'Saved CatBoost model to {path.with_suffix(".cbm")}')
    elif isinstance(model, BaseEstimator):
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print(f'Saved sklearn modelk to {path.with_suffix(".pkl")}')
    else:
        raise ValueError(f"Unsupported model type: {type(model)}.")
    

def calculate_prediction_metrics(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> dict:

    #Merge for comparison
    merged = pd.merge(predictions_df, actuals_df, on='datetime', how='inner')
    if merged.empty:
        return {
            'rmse': None,
            'mae': None,
            'num_predictions': 0,
            'num_actual': 0
        }
    mse = ((merged['prediction'] - merged['cnt']) ** 2).mean()
    rmse = mse ** 0.5 #sqrt of mse
    mae = abs(merged['prediction'] - merged['cnt']).mean()

    return {
            'rmse': rmse,
            'mae': mae,
            'num_predictions': len(predictions_df),
            'num_actual': len(actuals_df),
            'num_matched': len(merged)
        }

def make_prediction_figures(
    df_prod: pd.DataFrame,
    df_pred: pd.DataFrame,
    parameters,
    config,
    lookback_hours,
    shared_xrange
):
    # Handle missing or empty predictions
    if df_pred is None or df_pred.empty:
        max_time = df_prod['datetime'].max()
        min_time = max_time - pd.Timedelta(hours=lookback_hours)
        df_prod = df_prod[(df_prod['datetime'] >= min_time) & (df_prod['datetime'] <= max_time)]
        fig1 = go.Figure()
        fig2 = go.Figure()
        fig1.add_trace(go.Scattergl(
            x=df_prod['datetime'],
            y=df_prod['cnt'],
            name='True Bike Count',
            mode='lines+markers',
            line=dict(color='#F08080', width=2),
            marker=dict(symbol='x', size=6)
        ))
        for param in parameters:
            if param in df_prod.columns:
                fig2.add_trace(go.Scattergl(
                    x=df_prod['datetime'],
                    y=df_prod[param],
                    name=param,
                    mode='lines'
                ))
        for fig in [fig1, fig2]:
            fig.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=40, b=20),
                height=330,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Value",
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                xaxis=dict(
                    title_font=dict(color=config['ui']['axis_font_color']),
                    tickfont=dict(color=config['ui']['axis_font_color'])
                ),
                yaxis=dict(
                    title_font=dict(color=config['ui']['axis_font_color']),
                    tickfont=dict(color=config['ui']['axis_font_color'])
                ),
                plot_bgcolor='#fff',
                paper_bgcolor='#fff'
            )
        prod_max_time = df_prod['datetime'].max()
        for fig in [fig1, fig2]:
            fig.add_vline(
                x=prod_max_time,
                line_width=3,
                line_dash="dash",
                line_color="lightgrey"
            )
        pred_max_time = prod_max_time + pd.Timedelta(hours=2)
        if shared_xrange and len(shared_xrange) == 2:
            fig1.update_xaxes(range=shared_xrange)
            fig2.update_xaxes(range=shared_xrange)
        else:
            fig1.update_xaxes(range=[min_time, pred_max_time])
            fig2.update_xaxes(range=[min_time, pred_max_time])
        return fig1, fig2

    # If predictions exist, plot both
    max_time = max(df_pred['datetime'].max(), df_prod['datetime'].max())
    min_time = max_time - pd.Timedelta(hours=lookback_hours)
    df_pred = df_pred[(df_pred['datetime'] >= min_time) & (df_pred['datetime'] <= max_time)]
    df_prod = df_prod[(df_prod['datetime'] >= min_time) & (df_prod['datetime'] <= max_time)]
    merged = pd.merge(
        df_pred[['datetime', 'prediction']],
        df_prod[['datetime', 'cnt']],
        on='datetime',
        how='outer'
    )
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig1.add_trace(go.Scattergl(
        x=merged['datetime'],
        y=merged['cnt'],
        name='True Bike Count',
        mode='lines+markers',
        line=dict(color='#F08080', width=2),
        marker=dict(symbol='x', size=6)
    ))
    fig1.add_trace(go.Scattergl(
        x=merged['datetime'],
        y=merged['prediction'],
        name='Predicted Bike Count',
        mode='lines+markers',
        line=dict(color='#1E8449', width=2),
        marker=dict(symbol='circle', size=8)
    ))
    for param in parameters:
        if param in df_prod.columns:
            fig2.add_trace(go.Scattergl(
                x=df_prod['datetime'],
                y=df_prod[param],
                name=param,
                mode='lines'
            ))
    for fig in [fig1, fig2]:
        fig.update_layout(
            template='plotly_white',
            margin=dict(l=40, r=40, t=40, b=20),
            height=330,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Value",
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            xaxis=dict(
                title_font=dict(color=config['ui']['axis_font_color']),
                tickfont=dict(color=config['ui']['axis_font_color'])
            ),
            yaxis=dict(
                title_font=dict(color=config['ui']['axis_font_color']),
                tickfont=dict(color=config['ui']['axis_font_color'])
            ),
            plot_bgcolor='#fff',
            paper_bgcolor='#fff'
        )
    prod_max_time = df_prod['datetime'].max()
    for fig in [fig1, fig2]:
        fig.add_vline(
            x=prod_max_time,
            line_width=3,
            line_dash="dash",
            line_color="lightgrey"
        )
    pred_max_time = df_pred['datetime'].max() + pd.Timedelta(hours=2)
    if shared_xrange and len(shared_xrange) == 2:
        fig1.update_xaxes(range=shared_xrange)
        fig2.update_xaxes(range=shared_xrange)
    else:
        fig1.update_xaxes(range=[min_time, pred_max_time])
        fig2.update_xaxes(range=[min_time, pred_max_time])
    return fig1, fig2

    
