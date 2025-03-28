import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = xgb.Booster()
model.load_model('xgboost_model.json')

single_player_data = {
    'minutes': 25, 
    'fieldGoalsMade': 5, 
    'fieldGoalsAttempted': 10, 
    'fieldGoalsPercentage': 0.5,
    'threePointersMade': 2, 
    'threePointersAttempted': 4, 
    'threePointersPercentage': 0.5, 
    'freeThrowsMade': 1, 
    'freeThrowsAttempted': 2, 
    'freeThrowsPercentage': 0.5, 
    'reboundsOffensive': 1, 
    'reboundsDefensive': 3, 
    'reboundsTotal': 4, 
    'assists': 2, 
    'steals': 1, 
    'blocks': 0, 
    'againstTeamSlug': 'magic', 
    'days_since_recent_game': 15
}

single_player_df = pd.DataFrame([single_player_data])

encoder = LabelEncoder()
single_player_df['againstTeamSlug'] = encoder.fit_transform(single_player_df['againstTeamSlug'])

features = [
    'minutes', 'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',
    'threePointersMade', 'threePointersAttempted', 'threePointersPercentage',
    'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
    'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 
    'steals', 'blocks', 'againstTeamSlug', 'days_since_recent_game'
]

X_single_player = single_player_df[features]

dtest = xgb.DMatrix(X_single_player)

pred_proba = model.predict(dtest)

print(f'Predicted probability of scoring above 10 points: {pred_proba[0]:.4f}')
