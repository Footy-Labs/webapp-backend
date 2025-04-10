import pandas as pd
import os
from dotenv import load_dotenv

# Import your custom modules
from wyscout_utils.wyscout_ETL import load_and_concatenate_player_excels, process_columns, map_simplified_position
from wyscout_utils.wyscout_metrics import cleaned_wyscout_mapping

# SQLAlchemy imports for Supabase integration
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.types import JSON

# Load the .env file from the current directory
load_dotenv()

# Retrieve credentials from environment variables
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME")

# Create the connection URL
url_object = URL.create(
    "postgresql+psycopg2",
    username=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASSWORD,
    host=SUPABASE_DB_HOST,
    port=5432,
    database=SUPABASE_DB_NAME
)
engine = create_engine(url_object)

# ===================== 1) Load & Process Player Excel Files =====================
print("Loading player Excel files...")
players_match_data = load_and_concatenate_player_excels(directory_path=r"Data\Player Match Data")
print(f"Loaded {players_match_data.shape[0]} rows and {players_match_data.shape[1]} columns.\n")

# Loading A lyga Player General Data
players_general_data = load_and_concatenate_player_excels(directory_path=r"Data\Player General Data")

if players_general_data.shape[0] < 150:
    print(f"❌ Not enough data: only {players_general_data.shape[0]} rows found. Minimum required is 150.")

# Create a mapping from 'Player' to 'Position' based on the general data
position_map = players_general_data.set_index("Player")["Position"]

# Overwrite the 'Position' column in the match data using this mapping
players_match_data["Position"] = players_match_data["Player"].map(position_map)

print("Processing derived columns and mapping simplified positions...")
players_match_data = process_columns(players_match_data, cleaned_wyscout_mapping)
players_match_data = map_simplified_position(players_match_data)
print("Processing complete.\n")

print("Sample data after processing:")
print(players_match_data.sample(3))

# ===================== 1.5) Append New Percentile Columns and Merge Back =====================
six_metrics_with_legend = {
    'Goalkeeper': [
        ('Conceded goals per 90', 'low'),
        ('Accurate passes, %', 'high'),
        ('xG against per 90', 'low'),
        ('Prevented goals per 90', 'high'),
        ('Save rate, %', 'high'),
        ('Exits per 90', 'high')
    ],
    'Full Back': [
        ('Successful defensive actions per 90', 'high'),
        ('Defensive duels won, %', 'high'),
        ('Accurate crosses, %', 'high'),
        ('Accurate passes, %', 'high'),
        ('Key passes per 90', 'high'),
        ('xA per 90', 'high')
    ],
    'Centre Back': [
        ('Successful defensive actions per 90', 'high'),
        ('Defensive duels won, %', 'high'),
        ('Aerial duels won, %', 'high'),
        ('Interceptions per 90', 'high'),
        ('Accurate passes, %', 'high'),
        ('Accurate passes to final third per 90', 'high')
    ],
    'Defensive Midfielder': [
        ('Interceptions per 90', 'high'),
        ('Sliding tackles per 90', 'high'),
        ('Aerial duels won, %', 'high'),
        ('Accurate progressive passes per 90', 'high'),
        ('Accurate passes to final third per 90', 'high'),
        ('Accurate passes to penalty area per 90', 'high')
    ],
    'Central Midfielder': [
        ('Successful defensive actions per 90', 'high'),
        ('Defensive duels won, %', 'high'),
        ('Accurate passes, %', 'high'),
        ('Accurate passes to final third per 90', 'high'),
        ('Key passes per 90', 'high'),
        ('xA per 90', 'high')
    ],
    'Attacking Midfielder': [
        ('Defensive duels won, %', 'high'),
        ('Successful defensive actions per 90', 'high'),
        ('Accurate passes to penalty area per 90', 'high'),
        ('Accurate smart passes per 90', 'high'),
        ('Goals per 90', 'high'),
        ('Successful dribbles per 90', 'high')
    ],
    'Winger': [
        ('Non-penalty goals per 90', 'high'),
        ('xG per 90', 'high'),
        ('Shots on target per 90', 'high'),
        ('Successful dribbles per 90', 'high'),
        ('Assists per 90', 'high'),
        ('xA per 90', 'high')
    ],
    'Centre Forward': [
        ('Non-penalty goals per 90', 'high'),
        ('xG per 90', 'high'),
        ('Shots on target per 90', 'high'),
        ('Touches in box per 90', 'high'),
        ('xA per 90', 'high'),
        ('Offensive duels won, %', 'high')
    ]
}

def compute_ranking_for_position(players: pd.DataFrame, position: str, metric_pairs: list) -> pd.DataFrame:
    pos_subset = players.loc[players['Simplified Position'] == position].copy()
    # Work only on rows with complete data for these metrics
    complete_idx = pos_subset.dropna(subset=[mp[0] for mp in metric_pairs]).index
    ranking_df = pd.DataFrame(index=complete_idx)
    for metric_name, direction in metric_pairs:
        if direction == 'high':
            ranking_df[f'{metric_name}_percentile'] = pos_subset.loc[complete_idx, metric_name].rank(ascending=True, pct=True)
        else:
            ranking_df[f'{metric_name}_percentile'] = pos_subset.loc[complete_idx, metric_name].rank(ascending=False, pct=True)
    percentile_cols = [f'{m[0]}_percentile' for m in metric_pairs]
    ranking_df['avg_percentile'] = ranking_df[percentile_cols].mean(axis=1)
    return ranking_df

def add_ranking_columns_to_full_df(players: pd.DataFrame, six_metrics_with_legend: dict) -> pd.DataFrame:
    players_with_rank = players.copy()
    for position, metric_pairs in six_metrics_with_legend.items():
        ranking_df = compute_ranking_for_position(players, position, metric_pairs)
        # Update the original DataFrame with these new ranking columns for rows in this position
        for col in ranking_df.columns:
            players_with_rank.loc[ranking_df.index, col] = ranking_df[col]
    return players_with_rank

players_match_data = add_ranking_columns_to_full_df(players_match_data, six_metrics_with_legend)
print("Merged ranked DataFrame sample:")
print(players_match_data.sample(5))
print("Final DataFrame shape after merging ranking columns:", players_match_data.shape)

# ===================== 2) Retrieve Clubs Mapping from Supabase =====================
print("Retrieving clubs mapping from Supabase...")
with engine.connect() as conn:
    clubs_df = pd.read_sql("SELECT id, name FROM clubs;", conn)
clubs_map = dict(zip(clubs_df['name'], clubs_df['id']))
print("Clubs mapping retrieved:")
print(clubs_map)

# ===================== 3) Build Insertion DataFrame =====================
print("Building insertion DataFrame for 'players' table...")

def clean_nan(val):
    return None if pd.isna(val) else val

def build_full_stats_dict(row):
    # Convert the entire row to a dict, cleaning NaN values, without excluding any columns.
    return {k: clean_nan(v) for k, v in row.to_dict().items()}

insert_df = pd.DataFrame()
insert_df["name"] = players_match_data["Player"]
insert_df["club_id"] = players_match_data["Team"].apply(lambda team: clubs_map.get(team))
insert_df["position"] = players_match_data["Simplified Position"]
insert_df["stats"] = players_match_data.apply(build_full_stats_dict, axis=1)

print("Built insertion DataFrame. Sample:")
print(insert_df.head())
print("Final insertion DataFrame shape:", insert_df.shape)

# ===================== 4) Insert Data into Existing 'players' Table =====================
TABLE_NAME = "players"

print(f"Inserting data into '{TABLE_NAME}' table with if_exists='append'...")
insert_df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="append",
    index=False,
    dtype={"stats": JSON()}
)
print(f"Data inserted successfully into '{TABLE_NAME}' table!")

