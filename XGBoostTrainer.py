import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

cleaned = pd.read_csv('./cleaned/trimmed_final_with_pgs.csv')

def XGBoost(df: pd.DataFrame, threshold=10):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, seed=23, verbosity=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    residuals = y_test - y_pred

    residual_std_dev = np.std(residuals)
    print(f'Standard Dev: {residual_std_dev:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R2: {r2:.4f}')

    path = './cleaned/xgboost_model_new.json'
    model.save_model(path)
    print(f'Model saved to {path}')


XGBoost(cleaned)