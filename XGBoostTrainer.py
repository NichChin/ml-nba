import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

cleaned = pd.read_csv('./cleaned/final.csv')

def XGBoost(df: pd.DataFrame, threshold=10):
    df['game_date'] = pd.to_datetime(df['game_date'])

    most_recent_game = df['game_date'].max()

    df['days_sice_most_recent_game'] = (most_recent_game - df['game_date']).dt.days
    df = df.drop(columns='game_date')

    df['target'] = (df['points'] > threshold).astype(int)

    X = df.drop(columns=['target', 'points'])
    encoder = LabelEncoder()
    
    X['againstTeamSlug'] = encoder.fit_transform(X['againstTeamSlug'])
    X['teamSlug'] = encoder.fit_transform(X['teamSlug'])

    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

    path = './cleaned/xgboost_model.json'
    model.save_model(path)
    print(f'Model saved to {path}')


XGBoost(cleaned)