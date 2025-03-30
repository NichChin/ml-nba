from random import randrange
import time
import math
import concurrent
import pandas

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import commonteamroster

class GameScorePopulate:
    def __init__(self):
        self.slugToTeamId = {
            "hawks": 1610612737,  # Atlanta Hawks
            "celtics": 1610612738,  # Boston Celtics
            "nets": 1610612751,  # Brooklyn Nets
            "hornets": 1610612766,  # Charlotte Hornets
            "bobcats": 1610612766,  # Charlotte Bobcats
            "bulls": 1610612741,  # Chicago Bulls
            "cavaliers": 1610612739,  # Cleveland Cavaliers
            "mavericks": 1610612742,  # Dallas Mavericks
            "nuggets": 1610612743,  # Denver Nuggets
            "pistons": 1610612765,  # Detroit Pistons
            "warriors": 1610612744,  # Golden State Warriors
            "rockets": 1610612745,  # Houston Rockets
            "pacers": 1610612754,  # Indiana Pacers
            "clippers": 1610612746,  # Los Angeles Clippers
            "lakers": 1610612747,  # Los Angeles Lakers
            "grizzlies": 1610612763,  # Memphis Grizzlies
            "heat": 1610612748,  # Miami Heat
            "bucks": 1610612749,  # Milwaukee Bucks
            "timberwolves": 1610612750,  # Minnesota Timberwolves
            "pelicans": 1610612740,  # New Orleans Pelicans
            "knicks": 1610612752,  # New York Knicks
            "thunder": 1610612760,  # Oklahoma City Thunder
            "magic": 1610612753,  # Orlando Magic
            "sixers": 1610612755,  # Philadelphia 76ers
            "suns": 1610612756,  # Phoenix Suns
            "blazers": 1610612757,  # Portland Trail Blazers
            "kings": 1610612758,  # Sacramento Kings
            "spurs": 1610612759,  # San Antonio Spurs
            "raptors": 1610612761,  # Toronto Raptors
            "jazz": 1610612762,  # Utah Jazz
            "wizards": 1610612764  # Washington Wizards
        }

        self.playerCache = {} # (playerId, seasonStr) : gameScore
        self.teamCache = {} # (teamId, seasonStr) : gameScore

    def get_player_stats(self, player_id: int, season_year: int) -> list | None:
        """
        :param player_id: player's ID (e.g. 2544 for LeBron James).
        :param season_year: The last two digits of the end year of the season (e.g. 18 for 2017-2018 season).
        :return: Nullable list of the sum of each stat for a season. None if player is in their first season.
        """
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            if not career:
                return None

            season_stats = career.get_dict()['resultSets'][0]['rowSet']

            # Player is in their first season.
            if len(season_stats) < 2:
                return None
            else:
                # Get the stats for the season before
                for season in season_stats:
                    # Linear scan for the season before.
                    if int(season[1].split('-')[1]) == season_year - 1:
                        return season

        except Exception as ex:
            print("Player retrieval failed", ex)

    def get_player_game_score(self, season_stats) -> float | None:
        """
        :param season_stats: List of season stats.
        :return: Nullable float. None if no games played.
        """
        player_id, season_id, league_id, team_id, team_abbreviation, player_age, gp, gs, min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk, tov, pf, pts = season_stats
        if gp == 0:
            return None
        game_score = (pts + 0.4 * fgm - 0.7 * fga - 0.4 * (
                    fta - ftm) + 0.7 * oreb + 0.3 * dreb + stl + 0.7 * ast + 0.7 * blk - 0.4 * pf - tov) / gp
        return game_score

    def get_team_roster(self, team_name: str, season_str: str) -> list | None:
        """
        :param team_name: Team name string, like 'celtics', 'sixers'...
        :param season_str: Season year string, like '2017-18'
        :return:
        """
        try:
            roster = commonteamroster.CommonTeamRoster(self.slugToTeamId[team_name], season_str)
            if not roster:
                return None
            players = roster.get_dict()['resultSets'][0]['rowSet']
            return players

        except Exception as ex:
            print("Team retrieval failed", ex)

    def get_player_game_score_from_cache(self, player_id: int, season_str: str) -> float | None:
        """
        :param player_id: Player's ID (e.g. 2544 for LeBron James).'
        :param season_str: Season year string, like '2017-18'
        :return: A nullable float representing the player's average game score for that season.
        """
        if (player_id, season_str) in self.playerCache:
            return self.playerCache[(player_id, season_str)]
        return None

    def get_team_game_score_from_cache(self, team_slug: str, season_str: str) -> float | None:
        """
        :param team_slug: Team name string, like 'celtics', 'sixers'...
        :param season_str: Season year string, like '2017-18'
        :return: A nullable float representing the team's average game score for that season.
        """
        if (team_slug, season_str) in self.teamCache:
            return self.teamCache[(team_slug, season_str)]
        return None

    def calculate_team_game_score(self, player_game_scores: list) -> float | None:
        if len(player_game_scores) < 5:
            return None

# Execution block
# def fetch_player_stats(row, seen_players):
#     player_key = (row.personId, row.season_year)
#     if player_key not in seen_players:
#         sleep(2)
#         player_stats = populate.get_player_stats(row.personId, int(row.season_year.split('-')[1]))
#         if player_stats:
#             player_game_score = populate.get_player_game_score(player_stats)
#             seen_players[player_key] = player_game_score
#         else:
#             seen_players[player_key] = None
#     return seen_players[player_key]

# populate = GameScorePopulate()
# df = pandas.read_csv('./cleaned/final_with_pgs.csv')
#
# for row in df.itertuples():
#     if (row.personId, row.season_year) not in populate.playerCache:
#         populate.playerCache[(row.personId, row.season_year)] = row.playerGameScore
#
# i = 0
# for row in df.iloc[i:].itertuples():
#     team_score = populate.get_team_game_score_from_cache(row.teamSlug, row.season_year)
#     against_team_score = populate.get_team_game_score_from_cache(row.againstTeamSlug, row.season_year)
#
#     if not team_score:
#         team_score = 0
#         time.sleep(4)
#         player_list = populate.get_team_roster(row.teamSlug, row.season_year)
#         valid_players = 0
#
#         for player in player_list:
#             player_id = player[14]
#             player_score = populate.get_player_game_score_from_cache(player_id, row.season_year)
#
#             if player_score:
#                 team_score += player_score
#                 valid_players += 1
#
#         team_score = team_score / valid_players if valid_players > 5 else None
#
#     if not against_team_score:
#         against_team_score = 0
#         time.sleep(4)
#         against_player_list = populate.get_team_roster(row.againstTeamSlug, row.season_year)
#         valid_players = 0
#
#         for player in against_player_list:
#             player_id = player[14]
#             player_score = populate.get_player_game_score_from_cache(player_id, row.season_year)
#
#             if player_score:
#                 against_team_score += player_score
#                 valid_players += 1
#
#         against_team_score = against_team_score / valid_players if valid_players > 5 else None
#
#     populate.teamCache[(row.teamSlug, row.season_year)] = team_score
#     populate.teamCache[(row.againstTeamSlug, row.season_year)] = against_team_score
#
#     # with open('team_scores.txt', 'a') as file:
#     #     file.write(str(team_score) + '\n')
#     #
#     # with open('against_team_scores.txt', 'a') as file:
#     #     file.write(str(against_team_score) + '\n')
#
#     print(row.teamSlug, row.againstTeamSlug, i, team_score, against_team_score)
#     i += 1

# i = 0
# for row in df.iloc[i:].itertuples():
#     player_score = fetch_player_stats(row, seen_players)
#     with open('out.txt', 'a') as file:
#         file.write(str(player_score) + '\n')
#     print(i, player_score)
#     i += 1

# final_df = pandas.read_csv('./final.csv')
# final_df['playerGameScore'] = player_game_scores
#
# final_df.to_csv('./new_final.csv')