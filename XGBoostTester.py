import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from RandomForestTester import generate_test_data
from constants import *
import scipy.stats as stats

model = xgb.Booster()
model.load_model('./cleaned/xgboost_model_new.json')
playerNames = pd.read_csv('./cleaned/personNames.csv')

test_data = {
        'percent_completed': [0.9],
        'personId': [LEBRON_JAMES],
        'minutes': [40],  
        'playerGameScore': [20],  
        'teamGameScore': [7.8],  
        'againstTeamGameScore': [8.5],  
        'days_since_last_game': [1]
    }

def predict_single_player(player_df: pd.DataFrame, model=model):
    dtest = xgb.DMatrix(player_df)
    prediction = model.predict(dtest)

    playerId = player_df['personId'].iloc[0]
    playerName = playerNames[playerNames['personId'] == playerId]['personName'].values[0]

    print(f'{playerName} will score {prediction[0]} points')
    return prediction[0]

def compute_probability(predicted_points, threshold, under=False):
    prob_under = stats.norm.cdf(threshold, loc=predicted_points, scale=STD_DEV_XGBOOST)

    return prob_under if under else 1 - prob_under


pts = predict_single_player(generate_test_data(team_slug='lakers', against_team_slug='rockets', test_data=test_data))
print(compute_probability(pts, threshold=20, under=False))