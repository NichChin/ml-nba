import pandas as pd

team_slugs = pd.read_csv('./cleaned/team-slugs.csv')
team_slugs = dict(zip(team_slugs['teamTricode'], team_slugs['teamSlug']))
print(team_slugs)

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
    cols_to_drop = ['comment', 'matchup', 'season_year', 'jerseyNum', 'gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'personName', 'position', 'turnovers', 'foulsPersonal', 'plusMinusPoints']

    df_cleaned = df_cleaned.drop(cols_to_drop, axis=1)
    return df_cleaned

for i in range(1, 4):
    if i == 1:
        biggest_cleaned = cleaner(pd.read_csv('./NBA-Data-2010-2024/regular_season_box_scores_2010_2024_part_1.csv'))
    else:
        biggest_cleaned = pd.concat([biggest_cleaned, cleaner(pd.read_csv(f"./NBA-Data-2010-2024/regular_season_box_scores_2010_2024_part_{i}.csv"))], ignore_index=True)

biggest_cleaned.to_csv('./cleaned/final.csv', index=False)