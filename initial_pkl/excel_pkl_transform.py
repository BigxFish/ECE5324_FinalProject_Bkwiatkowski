import pandas as pd
import numpy as np
import pickle


#Import CSV file into a DataFrame
csv_file = 'spreadspoke_scores.csv'  # Replace with your actual file path
df = pd.read_csv(csv_file)

#Set the first column as the row index
df.set_index(df.columns[0], inplace=True)  # Uses the first column as the row index


#Dictionary mapping full names to abbreviations
nfl_team_abbreviations = {
    # Current Teams
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Historic Teams
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LAR",
    "Oakland Raiders": "LVR",
    "Houston Oilers": "TEN",
    "Baltimore Colts": "IND",
    "Boston Patriots": "BOS",
    "St. Louis Cardinals": "STL"
}


#Apply the mapping to both columns
df["team_home"] = df["team_home"].map(nfl_team_abbreviations)
df["team_away"] = df["team_away"].map(nfl_team_abbreviations)

#Display the updated DataFrame

df['result_spread'] = df['score_home'] - df['score_away']

df['spread_favorite'] = df.apply(
    lambda row: -row['spread_favorite'] if row['team_favorite_id'] == row['team_away'] else row['spread_favorite'],
    axis=1
)


columns_to_remove = ['schedule_playoff', 'schedule_season','schedule_week','over_under_line','stadium','stadium_neutral','team_away','team_favorite_id','team_home','weather_humidity','weather_detail']  # Replace with column names to remove
df = df.drop(columns=columns_to_remove)

#Save the modified DataFrame as a .pkl file
pkl_file = 'data.pkl'
with open(pkl_file, 'wb') as file:
    pickle.dump(df, file)

#Reopen the .pkl file
with open(pkl_file, 'rb') as file:
    loaded_df = pickle.load(file)
    
    
def convert_columns_to_float64(df):
    # Iterate over columns and convert to float64 if possible
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('float64')
    return df

convert_columns_to_float64(df=df)


print(df.head(5))




