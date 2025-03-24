import pandas as pd

def cleaner(df: pd.DataFrame):
    # remove rows with comments as the player did not play
    df_cleaned = df[df['comment'].isna()]

    # remove rows for players with less than 10 minutes played
    def convert_minutes_to_float(minutes_str):
        minutes, seconds = minutes_str.split(':')
        return int(minutes) + int(seconds) / 60

    df_cleaned['minutes'] = df_cleaned['minutes'].apply(convert_minutes_to_float)
    df_cleaned = df_cleaned[df_cleaned['minutes'] >= 10]

    # drop useless (unused) columns
    cols_to_drop = ['comment', 'jerseyNum', 'matchup', 'season_year', 'game_date', 'gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'personName', 'position', 'turnovers', 'foulsPersonal', 'plusMinusPoints']
    
    # TODO: before deleting matchup, split and add column for against team (againstTeamSlug)
    # TODO: re-add game date
    df_cleaned = df_cleaned.drop(cols_to_drop, axis=1)
    return df_cleaned

for i in range(1, 4):
    if i == 1:
        biggest_cleaned = cleaner(pd.read_csv('./NBA-Data-2010-2024/regular_season_box_scores_2010_2024_part_1.csv'))
    else:
        biggest_cleaned = pd.concat([biggest_cleaned, cleaner(pd.read_csv(f"./NBA-Data-2010-2024/regular_season_box_scores_2010_2024_part_{i}.csv"))], ignore_index=True)

biggest_cleaned.to_csv('./cleaned/final.csv', index=False)