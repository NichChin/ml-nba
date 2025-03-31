import pandas as pd
import datetime

team_slugs = pd.read_csv('./cleaned/team-slugs.csv')
team_slugs = dict(zip(team_slugs['teamTricode'], team_slugs['teamSlug']))
# print(team_slugs)

def cleaner(df: pd.DataFrame):
    # remove rows with comments as the player did not play
    df_cleaned = df[df['comment'].isna()]

    # remove rows for players with less than 10 minutes played
    def convert_minutes_to_float(minutes_str):
        minutes, seconds = minutes_str.split(':')
        return int(minutes) + int(seconds) / 60

    df_cleaned['minutes'] = df_cleaned['minutes'].apply(convert_minutes_to_float)
    df_cleaned = df_cleaned[df_cleaned['minutes'] >= 10]

    def add_against_slug(matchup_str: str):
        against = matchup_str.split()[-1]
        against_slug = team_slugs[against]
        return against_slug

    df_cleaned['againstTeamSlug'] = df_cleaned['matchup'].apply(add_against_slug)

    # drop useless (unused) columns
    cols_to_drop = ['comment', 'matchup', 'jerseyNum', 'gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'position', 'turnovers', 'foulsPersonal', 'plusMinusPoints']

    df_cleaned = df_cleaned.drop(cols_to_drop, axis=1)
    return df_cleaned

def final_clean(df: pd.DataFrame):
    """
    from newly created dataset, add a column for percentage of season complete
    """
    df['game_date'] = pd.to_datetime(df['game_date'])

    def add_percent_season_complete(season_year: str, game_date: datetime.datetime):
        season_start_year = int(season_year.split('-')[0])
        season_start_date = datetime.datetime(year=season_start_year, month=10, day=10)
        season_end_date = datetime.datetime(year=season_start_year + 1, month=4, day=30)
        return (game_date - season_start_date) / (season_end_date - season_start_date)
    
    df['percent_completed'] = df[['season_year', 'game_date']].apply(lambda row: add_percent_season_complete(row[0], row[1]), axis=1)
    
    df = df[['game_date', 'percent_completed', 'teamSlug', 'personId', 'againstTeamSlug', 'minutes', 'playerGameScore', 'teamGameScore', 'againstTeamGameScore', 'points']]
    df.to_csv('./cleaned/trimmed_final_with_pgs.csv', index=False)

final_clean(pd.read_csv('./cleaned/final_with_pgs.csv'))
# for i in range(1, 4):
#     if i == 1:
#         biggest_cleaned = cleaner(pd.read_csv('./NBA-Data-2010-2024/regular_season_box_scores_2010_2024_part_1.csv'))
#     else:
#         biggest_cleaned = pd.concat([biggest_cleaned, cleaner(pd.read_csv(f"./NBA-Data-2010-2024/regular_season_box_scores_2010_2024_part_{i}.csv"))], ignore_index=True)

# biggest_cleaned.to_csv('./cleaned/final.csv', index=False)

# def for_rc(df: pd.DataFrame):
#     df = df[['season_year', 'game_date', 'teamSlug', 'personId', 'againstTeamSlug']]
#     df.to_csv('./cleaned/for_rc.csv', index=False)
#
# for_rc(pd.read_csv('./cleaned/final.csv'))