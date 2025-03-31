import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# df = pd.read_csv('./cleaned/final.csv')
# model = joblib.load('./models/random_forest_model.pkl')

def evaluate_test_set(df, model, test_size = 0.2):
    """
    Splits the data, evaluates the model on the split test set, and prints metrics.
    df: The full dataset.
    model: The trained model.
    test_size: The proportion of the data used for testing.
    """
    # Splitting the dataset
    target = df.pop('points')
    features = df
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=42
    )

    # Preprocessing and predictions
    X_test_processed = model.named_steps['preprocessor'].transform(X_test)
    predictions = model.named_steps['regressor'].predict(X_test_processed)

    # Evaluate predictions and print metrics
    mse = mean_squared_error(y_test, predictions)
    mad = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Deviation (MAD): {mad:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R² Score: {r_squared:.4f}")


def compute_std_dev(df: pd.DataFrame, model):
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


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23)

    # predict and evaluate the model
    y_pred = model.predict(X_test)

    residuals = y_test - y_pred

    # compute the standard deviation of the residuals
    residual_std_dev = np.std(residuals)
    print(f'Standard Deviation of Residuals: {residual_std_dev}')

compute_std_dev(pd.read_csv('./cleaned/trimmed_final_with_pgs.csv'), model=joblib.load('./models/random_forest_model_new.pkl'))
# evaluate_test_set(df, model)