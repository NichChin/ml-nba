import pandas as pd
import joblib
import os
from constants import *

# def calculate_probability(model, X, threshold):
#     predicted_points = model.predict(X)
    
#     probabilities = (predicted_points > threshold).astype(int)
    
#     return probabilities

cleaned = pd.read_csv('./cleaned/final.csv')

def predict_single_player(player_id, against_team_slug, df, model):
    player_data = df[df['personId'] == player_id]

    if player_data.empty:
        print(f"No data found for playerId {player_id}.")
        return

    avg_stats = player_data[NUMERIC_FEATURES].mean().to_frame().T

    X_pred = pd.DataFrame({
        'personId': [player_id],
        'againstTeamSlug': [against_team_slug],
        'teamSlug': player_data['teamSlug'].iloc[0]
    })

    X_pred_with_avg_stats = X_pred.join(avg_stats)

    X_pred_with_avg_stats = X_pred_with_avg_stats.drop(columns=['personId'])

    predicted_points = model.predict(X_pred_with_avg_stats)

    print(f"Predicted points for player {player_id} against team {against_team_slug}: {predicted_points[0]}")

model = joblib.load('./models/random_forest_model.pkl')

player_id = LEBRON_JAMES  # Example playerId
against_team_slug = 'hornets'  # Example matchup team slug

predict_single_player(player_id, against_team_slug, cleaned, model)