# Group 7 MSE446 Term Project

## ML Methods Used

Random Forest Regressors and XGBoost Regressors were used to predict the number of points a player scores based on the matchup (`team`, and `againstTeam`), the relevant game scores (personal, team, and against), `percent_completed`, and `days_since_last_game`. 

## Assumptions

The models trained were regressors, but the problem being solved is one of classification—as a result, a significant assumptions was necessary to convert the predicted points scored to a probability the player scores over/under a threshold. The assumption is as follows: the distribution of points follows a normal distribution with mean `predicted_points` (standard deviation was calculated experimentally as seen in `compute_std_dev` in Measurements.py.

In terms of basketball assumptions, many simplifications were made. Some examples of simplifications are below:

- The last season's game score is indicative of their performance this season
- The roster stays consistent (relevant for each team's game score)
- No injuries (which would affect the player's performance in future games)


## Dataset Description

Data from regular season games from 2010-2024 was used to train the model on matchup and player rosters. It includes the following:

```
{season_year,game_date,teamSlug,personId,personName,minutes,fieldGoalsMade,fieldGoalsAttempted,fieldGoalsPercentage,threePointersMade,threePointersAttempted,threePointersPercentage,freeThrowsMade,freeThrowsAttempted,freeThrowsPercentage,reboundsOffensive,reboundsDefensive,reboundsTotal,assists,steals,blocks,points,againstTeamSlug,playerGameScore,teamGameScore,againstTeamGameScore}
```

More features were needed beyond what was provided in the `NBA-Data-2010-2024` dataset (provided by the subrepo), so `playerGameScore`, `teamGameScore`, `againstTeamGameScore` were calculated from data collected by the nba-api (details in `PlayerGameScore.py`).

## Model Performance

| Metric                              | Random Forest               | XGBoost                    |
|-------------------------------------|-----------------------------|----------------------------|
| **Mean Squared Error (MSE)**        | 28.4729                     | 26.9709                    |
| **Mean Absolute Deviation (MAD)**   | 4.1161                      | 3.9983                     |
| **Mean Absolute Percentage Error (MAPE)** | 950384845960856.1250   | 900276617740288.0000       |
| **R² Score**                        | 0.5729                      | 0.5899                     |
| **Standard Deviation of Residuals** | 5.3357                      | 5.1933                     |

As some player's predicted points are 0 (or close to 0), the MAPE is arbitrarily large. 

## File Structure

`./cleaned/` -> contains the cleaned data  
`./cleaned/encoded_col_names.csv` -> has the encoded column names, used to create the test data  
`./cleaned/final_with_pgs.csv` -> has all the relevant data from the `NBA-Data-2010-2024` dataset in one file, in addition to the relevant game scores  
`./cleaned/trimmed_final_with_pgs.csv` -> data used to train/test the model  
`./models/` -> contains the trained models. The suffix _initial indicates this was the initial model trained where there was excessive data leakage (resulting in an R^2 score of 1). Conversely, _new denotes the newer model, without the data leakage.  

Note: to test the random forest model, `random_forest_model_new.7z` will have to be unzipped. If errors occur with pulling, please contact Nicholas.  

## Group Members

Ronald Chen, Nicholas Chin, Derek Gao, Martin Lee
