import pandas as pd

def summary():
    bigsummary = pd.read_csv('./cleaned/final.csv')
    bigsummary['game_date'] = pd.to_datetime(bigsummary['game_date'])
    print(bigsummary['game_date'][1])

summary()