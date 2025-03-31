# Group 7 MSE446 Term Project

## ML Methods Used:

The ensemble methods Random Forest and XGBoost were used to predict the number of points a player scores based on the matchup (`team`, and `againstTeam`), and the relevant game scores (personal, team, and against).

## Data Collection

Data from regular season games from 2010-2024 was used to train the model on matchup and player rosters. It includes the following:

```
{season_year,game_date,teamSlug,personId,personName,minutes,fieldGoalsMade,fieldGoalsAttempted,fieldGoalsPercentage,threePointersMade,threePointersAttempted,threePointersPercentage,freeThrowsMade,freeThrowsAttempted,freeThrowsPercentage,reboundsOffensive,reboundsDefensive,reboundsTotal,assists,steals,blocks,points,againstTeamSlug,playerGameScore,teamGameScore,againstTeamGameScore}
```

More features were needed beyond what was provided in the `NBA-Data-2010-2024` dataset, so `playerGameScore`, `teamGameScore`, `againstTeamGameScore` were calculated from data collected by the nba-api (details in `PlayerGameScore.py`).

## TODO

Add metrics from model once trained

## Group Members

Ronald Chen, Nicholas Chin, Derek Gao, Martin Lee
