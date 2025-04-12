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

# ===================== 3) Build Insertion DataFrame =====================
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

print("Built insertion DataFrame. Sample:")
print(insert_df.head(), "\n")
print("Final insertion DataFrame shape:", insert_df.shape)

# ===================== 4) Insert into Supabase =====================
TABLE_NAME = "team_match_stats"

print(f"Inserting data into '{TABLE_NAME}' table with if_exists='append'...")
insert_df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="append",
    index=False,
    dtype={"stats": JSON()}
)
print(f"✅ Data inserted successfully into '{TABLE_NAME}'!")
