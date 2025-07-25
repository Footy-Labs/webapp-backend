import sqlalchemy
from dotenv import load_dotenv

# Import your custom modules
from wyscout_utils.wyscout_ETL import *
from wyscout_utils.wyscout_metrics import *
from wyscout_utils.wyscout_plots import *
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

print("Retrieving clubs mapping from Supabase...")
with engine.connect() as conn:
    clubs_df = pd.read_sql("SELECT id, name, league FROM clubs;", conn)
# Create a map from club name to club ID
club_to_id_map = dict(zip(clubs_df['name'], clubs_df['id']))
print("Club to ID mapping retrieved.")

# Create a map from club name to league name
club_to_league_map = dict(zip(clubs_df['name'], clubs_df['league']))
print("Club to League mapping retrieved.")
################# Lithuanian A Lyga part ###########################################
# ===================== 1) Load & Process Player Excel Files =====================
print("Loading player Excel files...")
lithuanian_players_match_data = load_and_concatenate_player_excels(directory_path=r"Data/Player Match Data/Lithuania")
print(f"Loaded {lithuanian_players_match_data.shape[0]} rows and {lithuanian_players_match_data.shape[1]} columns.\n")

# Adding the league as that will be important for the "avg_percentile" column later
lithuanian_players_match_data['League'] = lithuanian_players_match_data['Team'].map(club_to_league_map)

# Loading A lyga Player General Data
lithuanian_players_general_data = load_and_concatenate_player_excels(directory_path=r"Data/Player General Data/Lithuania")

if lithuanian_players_general_data.shape[0] < 150:
    print(f"❌ Not enough data: only {lithuanian_players_general_data.shape[0]} rows found. Minimum required is 150.")

# Create a mapping from 'Player' to 'Position' based on the general data
lithuanian_position_map = lithuanian_players_general_data.set_index("Player")["Position"]

# Overwrite the 'Position' column in the match data using this mapping
lithuanian_players_match_data["Position"] = lithuanian_players_match_data["Player"].map(lithuanian_position_map)

print("Processing derived columns and mapping simplified positions...")
lithuanian_players_match_data = process_columns(lithuanian_players_match_data, cleaned_wyscout_mapping)
lithuanian_players_match_data = map_simplified_position(lithuanian_players_match_data)
print("Processing complete.\n")

# Load the contract update file
lithuanian_contract_updates = pd.read_excel(r"Data/Transfermarkt Player Data/Lithuania/contract_info.xlsx")

# Clean player names
lithuanian_players_match_data["Player"] = lithuanian_players_match_data["Player"].str.strip()
lithuanian_contract_updates["Player"] = lithuanian_contract_updates["Player"].str.strip()

# Create a mapping from player to updated contract date
lithuanian_contract_map = dict(zip(lithuanian_contract_updates["Player"], lithuanian_contract_updates["Contract until"]))

# Apply the update only where there is a match
lithuanian_players_match_data["Contract expires"] = lithuanian_players_match_data.apply(
    lambda row: lithuanian_contract_map.get(row["Player"], row["Contract expires"]), axis=1)

print("Sample Lithuanian data after processing:")
print(lithuanian_players_match_data.sample(3))

####################################### Data processing for Lithuania done ###################

################# Latvian Virsliga  part ###########################################
# ===================== 1) Load & Process Player Excel Files =====================
print("Loading Latvian player Excel files...")
latvian_players_match_data = load_and_concatenate_player_excels(directory_path=r"Data/Player Match Data/Latvia")

print(f"Loaded {latvian_players_match_data.shape[0]} rows and {latvian_players_match_data.shape[1]} columns.\n")

# Adding the league as that will be important for the "avg_percentile" column later
latvian_players_match_data['League'] = latvian_players_match_data['Team'].map(club_to_league_map)

# Loading Latvian Virsliga Player General Data
latvian_players_general_data = load_and_concatenate_player_excels(directory_path=r"Data/Player General Data/Latvia")

if latvian_players_general_data.shape[0] < 150:
    print(f"❌ Not enough data: only {latvian_players_general_data.shape[0]} rows found. Minimum required is 150.")

# Create a mapping from ('Player', 'Team') to 'Position' based on the general data
latvian_position_map = latvian_players_general_data.set_index(['Player', 'Team'])["Position"]

# Create a temporary MultiIndex from the match data's Player and Team columns
temp_match_multi_index = pd.MultiIndex.from_frame(latvian_players_match_data[['Player', 'Team']])

# Overwrite/create the 'Position' column in the match data using this mapping
latvian_players_match_data["Position"] = temp_match_multi_index.map(latvian_position_map)

print("Processing derived columns and mapping simplified positions...")
latvian_players_match_data = process_columns(latvian_players_match_data, cleaned_wyscout_mapping)
latvian_players_match_data = map_simplified_position(latvian_players_match_data)
print("Processing complete.\n")

# # Load the contract update file from Transfermarkt
latvian_contract_updates = pd.read_excel(r"Data/Transfermarkt Player Data/Latvia/contract_info.xlsx")
# # Clean player names
latvian_players_match_data["Player"] = latvian_players_match_data["Player"].str.strip()

# # Create a mapping from player to updated contract date
latvian_contract_map = dict(zip(latvian_contract_updates["Player"], latvian_contract_updates["Contract until"]))

# # Apply the update only where there is a match
latvian_players_match_data["Contract expires"] = latvian_players_match_data.apply(
    lambda row: latvian_contract_map.get(row["Player"], row["Contract expires"]), axis=1)
#
print("Sample Latvian data after processing:")
print(latvian_players_match_data.sample(3))
################ Data processing for Latvia done ##################

all_players_match_data = pd.concat([lithuanian_players_match_data, latvian_players_match_data], ignore_index=True)

# ===================== 1.5) Append New Percentile Columns and Merge Back =====================
def add_ranking_columns_to_full_df(master_players_df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    """
    Adds percentile ranking columns to the main DataFrame, calculated per league and per position.
    master_players_df: The combined DataFrame with data from all leagues.
    """
    if master_players_df.empty:
        print("Input DataFrame for ranking is empty. Returning as is.")
        return master_players_df

    players_with_rank = master_players_df.copy()

    # Initialize all potential new percentile columns with pd.NA
    all_new_cols = ['avg_percentile']
    for pos_metrics in metrics.values():
        for metric_name, _ in pos_metrics:
            all_new_cols.append(f'{metric_name}_percentile')
    print("all_new_cols are", all_new_cols)

    for new_col_name in set(all_new_cols):  # Use set to avoid duplicates
        if new_col_name not in players_with_rank.columns:
            players_with_rank[new_col_name] = pd.NA  # Initialize as float to hold NA or numbers

    # Store all calculated ranking DataFrames and merge at the end for efficiency
    all_rankings_list = []

    # Group by 'League', then process positions within each league
    for league_name, league_group_df in players_with_rank.groupby('League'):
        print(f"Calculating rankings for League: {league_name} ({len(league_group_df)} players)")
        for position, metric_pairs in metrics.items():
            # Filter for the current position WITHIN the current league's data
            pos_subset_in_league = league_group_df[league_group_df['Simplified Position'] == position].copy()
            if pos_subset_in_league.empty:
                print(f"No players in League '{league_name}' for Position '{position}'. Skipping.")
                continue
            print("running code for position ", position, "in league ", league_name)
            ranking_df_for_pos_league = compute_ranking_for_position(pos_subset_in_league, metric_pairs)

            if not ranking_df_for_pos_league.empty:
                all_rankings_list.append(ranking_df_for_pos_league)

    if all_rankings_list:
        final_rankings_df = pd.concat(all_rankings_list)

        for col_to_update in final_rankings_df.columns:
            if col_to_update in players_with_rank.columns:
                # Use pd.to_numeric for safer conversion to float, especially with pd.NA
                players_with_rank[col_to_update] = pd.to_numeric(players_with_rank[col_to_update], errors='coerce')
                final_rankings_df[col_to_update] = pd.to_numeric(final_rankings_df[col_to_update], errors='coerce')
                players_with_rank.update(final_rankings_df[col_to_update])
            else:
                players_with_rank[col_to_update] = pd.to_numeric(final_rankings_df[col_to_update], errors='coerce')
    else:
        print("No ranking data generated across all leagues/positions to merge.")
    return players_with_rank


# After defining the functions, and after concatenating Lithuanian and Latvian data into 'all_players_match_data':
if not all_players_match_data.empty:
    print("\nStarting percentile calculations on combined data...")
    all_players_match_data_ranked = add_ranking_columns_to_full_df(all_players_match_data, six_metrics_with_legend)
    print("Percentile calculations complete.")
    print("Merged ranked DataFrame sample:")
    print(all_players_match_data_ranked.sample(5))
else:
    print("Combined data is empty, skipping ranking.")
    all_players_match_data_ranked = pd.DataFrame()  # Ensure it's an empty DF

# ===================== 3) Build Insertion DataFrame =====================
print("Building insertion DataFrame for 'players' table...")

def clean_nan(val):
    return None if pd.isna(val) else val

def build_full_stats_dict(row):
    # Convert the entire row to a dict, cleaning NaN values, without excluding any columns.
    return {k: clean_nan(v) for k, v in row.to_dict().items()}

insert_df = pd.DataFrame()
insert_df["name"] = all_players_match_data_ranked["Player"]
insert_df["club_id"] = all_players_match_data_ranked["Team"].apply(lambda team: club_to_id_map.get(team))
insert_df["position"] = all_players_match_data_ranked["Simplified Position"]
insert_df["stats"] = all_players_match_data_ranked.apply(build_full_stats_dict, axis=1)


# Extract wyscout id from stats json and put it in the dedicated wyscout_player_id column ---
print("Extracting Wyscout ID from stats...")
def get_wyscout_id(stats_dict):
    # Safely extracts the 'id' key from the stats dictionary
    if isinstance(stats_dict, dict):
        # Convert to int/None, handle potential non-numeric IDs gracefully if needed
        try:
            val = stats_dict.get('id', None)
            return int(val) if val is not None else None
        except (ValueError, TypeError):
             print(f"Warning: Could not convert Wyscout ID '{stats_dict.get('id')}' to int.")
             return None # Or handle as text if IDs aren't always numbers
    return None

# Apply this function to the 'stats' column we just created in insert_df
insert_df["wyscout_player_id"] = insert_df["stats"].apply(get_wyscout_id)
print("✅ Wyscout ID column added.")

print("Built insertion DataFrame. Sample:")
print(insert_df.head())
print("Final insertion DataFrame shape:", insert_df.shape)

# ===================== 4) Insert Data into Existing 'players' Table =====================
TABLE_NAME = "players"

print(f"Inserting data into '{TABLE_NAME}' table with if_exists='append'...")

insert_df.to_sql(TABLE_NAME, engine, if_exists="append", index=False, dtype={"stats": JSON()})

print(f"Data inserted successfully into '{TABLE_NAME}' table!")