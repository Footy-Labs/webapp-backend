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

# ===================== 1) Load Team Match Data =====================
print("Loading team match data...")
team_matches = load_and_concatenate_games_excels(r'Data\Team Match Data')
print(f"Loaded {team_matches.shape[0]} rows and {team_matches.shape[1]} columns.\n")

# ===================== 2) Retrieve Clubs Mapping from Supabase =====================
print("Retrieving clubs mapping from Supabase...")
with engine.connect() as conn:
    clubs_df = pd.read_sql("SELECT id, name FROM clubs;", conn)
clubs_map = dict(zip(clubs_df['name'], clubs_df['id']))
print("Clubs mapping retrieved:")
print(clubs_map, "\n")

# ===================== 3) Build Insertion DataFrame for Match-Level Stats =====================
def clean_nan(val):
    return None if pd.isna(val) else val

def build_stats_dict(row, exclude_cols):
    return {k: clean_nan(v) for k, v in row.to_dict().items() if k not in exclude_cols}

# Define which columns to preserve outside of stats
meta_cols = ["Match", "Date", "Competition", "Team"]
insert_df = pd.DataFrame()
insert_df["match_id"] = team_matches["Match"]
insert_df["date"] = pd.to_datetime(team_matches["Date"])
insert_df["competition"] = team_matches["Competition"]
insert_df["team_id"] = team_matches["Team"].map(clubs_map)
insert_df["stats"] = team_matches.apply(lambda row: build_stats_dict(row, meta_cols), axis=1)

print("Built match-level insertion DataFrame. Sample:")
print(insert_df.head(), "\n")
print("Final shape:", insert_df.shape)

# ===================== 4) Insert Match-Level Stats into Supabase =====================
MATCH_TABLE = "team_match_stats"

print(f"Inserting match-level data into '{MATCH_TABLE}'...")
insert_df.to_sql(
    MATCH_TABLE,
    engine,
    if_exists="replace",
    index=False,
    dtype={"stats": JSON()}
)
print(f"✅ Match-level data inserted into '{MATCH_TABLE}'!\n")

# ===================== 5) Build and Upload Aggregated Team-Level Stats =====================
print("Aggregating team metrics...")

# Exclude columns we don't want to average
exclude_cols = ["Duration", "Points Earned", "Match Day"]
numeric_cols = team_matches.select_dtypes(include='number').columns.tolist()
for col in exclude_cols:
    if col in numeric_cols:
        numeric_cols.remove(col)

# Aggregate
agg_dict = {col: 'mean' for col in numeric_cols}
agg_dict['Points Earned'] = 'sum'
team_metrics = team_matches.groupby('Team').agg(agg_dict).reset_index()

# Add team_id
team_metrics["team_id"] = team_metrics["Team"].map(clubs_map)
# Reorder to place team_id first
cols = ["team_id", "Team"] + [col for col in team_metrics.columns if col not in ["team_id", "Team"]]
team_metrics = team_metrics[cols]

print("Aggregated team-level stats:")
print(team_metrics.head(), "\n")
print("Final shape:", team_metrics.shape)

# ===================== 5.5) Merge Goal Distribution =====================
print("Merging goal distribution into team metrics...")

# Load goal distribution Excel (already saved earlier)
goal_dist_path = r"Data\Team Goal Distribution\Goal distribution A Lyga Teams 2025.xlsx"
df_goal_distribution = pd.read_excel(goal_dist_path)

# Make sure to rename "Team" index to a column if needed
if df_goal_distribution.index.name == "Team":
    df_goal_distribution = df_goal_distribution.reset_index()

team_metrics = team_metrics.merge(df_goal_distribution, on="Team", how="left")

print("✅ Goal distribution successfully merged into team_metrics.\n")

# ===================== 6) Insert Aggregated Stats =====================
AGG_TABLE = "team_metrics_aggregated"

print(f"Inserting aggregated data into '{AGG_TABLE}'...")
team_metrics.to_sql(
    AGG_TABLE,
    engine,
    if_exists="replace",
    index=False
)
print(f"✅ Aggregated team metrics inserted into '{AGG_TABLE}'!")
