import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from constants import *

cleaned = pd.read_csv('./cleaned/trimmed_final_with_pgs.csv')

def randomForestTrainer(df: pd.DataFrame):
    y = df.pop('points')

    df['game_date'] = pd.to_datetime(df['game_date'])

    most_recent_game = df['game_date'].max()
    df['days_since_last_game'] = (most_recent_game - df['game_date']).dt.days
    df = df.drop('game_date', axis=1)
    
    encoder = OneHotEncoder(sparse_output=False)
    
    categorical_cols = ['teamSlug', 'againstTeamSlug']
    encoded_cols = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    df = df.drop(categorical_cols, axis=1)
    X = pd.concat([df, encoded_df], axis=1)

    model = RandomForestRegressor(n_estimators=100, random_state=23, verbose=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23)

    model.fit(X_train, y_train)

    # predict and evaluate the model
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    
    model_directory = './models'
    # create dir if it doesn't exist
    os.makedirs(model_directory, exist_ok=True)

    model_filename = os.path.join(model_directory, 'random_forest_model_new.pkl')

    # export
    joblib.dump(model, model_filename)
    print(f'Model has been saved to {model_filename}')

# test(cleaned, joblib.load('./models/random_forest_model.pkl'))
randomForestTrainer(pd.read_csv('./cleaned/trimmed_final_with_pgs.csv'))