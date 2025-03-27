import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('./cleaned/final.csv')
model = joblib.load('./models/random_forest_model.pkl')

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

evaluate_test_set(df, model)