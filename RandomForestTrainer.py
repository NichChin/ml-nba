import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from constants import *

cleaned = pd.read_csv('./cleaned/final.csv')

def randomForestTrainer(df: pd.DataFrame):
    target = df.pop('points')

    # assign higher weight to most recent games
    df['game_date'] = pd.to_datetime(df['game_date'])
    most_recent_date = df['game_date'].max()

    df['days_since_game'] = (most_recent_date - df['game_date']).dt.days
    df['recency_weight'] = 1 / (df['days_since_game'] + 1)
    sample_weights = df['recency_weight']

    # standardize and encode num/categorical features respectively
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(), CATEGORICAL_FEATURES)
        ])

    # create pipeline to train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        df, target, sample_weights, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, regressor__sample_weight=weights_train)

    # predict and evaluate the model
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    model_directory = './models'

    # create dir if it doesn't exist
    os.makedirs(model_directory, exist_ok=True)

    model_filename = os.path.join(model_directory, 'random_forest_model.pkl')

    # export
    joblib.dump(model, model_filename)
    print(f'Model has been saved to {model_filename}')

# def calculate_probability(model, X, threshold):
#     predicted_points = model.predict(X)
    
#     probabilities = (predicted_points > threshold).astype(int)
    
#     return probabilities

def test(df, model):
    X_pred = df[['personId', 'againstTeamSlug', 'teamSlug']]

    avg_stats = df.groupby('personId')[NUMERIC_FEATURES].mean().reset_index()

    X_pred_with_avg_stats = X_pred.merge(avg_stats, on='personId', how='left')

    predictions = model.predict(X_pred_with_avg_stats.drop(columns=['personId']))

    print(f'Predicted points for each player in the test set: {predictions}')


test(cleaned, joblib.load('./models/random_forest_model.pkl'))
# randomForestTrainer(pd.read_csv('./cleaned/final.csv'))