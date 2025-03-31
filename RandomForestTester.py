import pandas as pd
import joblib
import numpy as np
from constants import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

cleaned = pd.read_csv('./cleaned/trimmed_final_with_pgs.csv')

def predict_single_player(player_df: pd.DataFrame, model: RandomForestRegressor) -> float:
    prediction = model.predict(player_df)

    print(prediction)
    return prediction[0]

model = joblib.load('./models/random_forest_model_new.pkl')

test_data = {
        'percent_completed': [0.5],
        'personId': [LEBRON_JAMES],
        'minutes': [35],  
        'playerGameScore': [25],  
        'teamGameScore': [7.8],  
        'againstTeamGameScore': [6.5],  
        'days_since_last_game': [5]  
    }

def generate_test_data(against_team_slug, team_slug, encoded_col_names_file='./cleaned/encoded_col_names.csv'):
    column_names = pd.read_csv(encoded_col_names_file, header=None).iloc[0].tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    teams = sorted(set([col.split('_')[1] for col in column_names if 'teamSlug' in col or 'againstTeamSlug' in col]))

    team_slug_encoded = encoder.fit_transform(np.array(teams).reshape(-1, 1))
    team_dict = {team: idx for idx, team in enumerate(teams)}
    
    team_slug_one_hot = [0] * len(teams)
    against_team_slug_one_hot = [0] * len(teams)
    
    if team_slug in team_dict:
        team_slug_one_hot[team_dict[team_slug]] = 1
    
    if against_team_slug in team_dict:
        against_team_slug_one_hot[team_dict[against_team_slug]] = 1

    for idx, team in enumerate(teams):
        test_data[f'teamSlug_{team}'] = [team_slug_one_hot[idx]]
        test_data[f'againstTeamSlug_{team}'] = [against_team_slug_one_hot[idx]]
    
    test_df = pd.DataFrame(test_data)
    
    test_df = test_df[column_names]

    return test_df

predict_single_player(player_df=generate_test_data('wizards', 'lakers'), model=model)