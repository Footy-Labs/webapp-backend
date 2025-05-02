from dotenv import load_dotenv

# Import your custom modules
from wyscout_utils.wyscout_ETL import *
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

# ===================== 1) Load Excel File =====================
print("Loading A Lyga final position averages table...")
file_path = r"Data\A Lyga Season Data\Final Position Table.xlsx"
df = pd.read_excel(file_path)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.\n")

# ===================== 2) Prepare Insert Table =====================
print("Preparing data for insertion...")

def clean_nan(val):
    return None if pd.isna(val) else val

def build_stats_dict(row):
    return {k: clean_nan(v) for k, v in row.to_dict().items() if k != "Position"}

insert_df = pd.DataFrame()
insert_df["position"] = df["Position"]
insert_df["stats"] = df.apply(build_stats_dict, axis=1)

print("Sample of insertion DataFrame:")
print(insert_df.head())

# ===================== 3) Push to Supabase =====================
TABLE_NAME = "final_position_averages"

print(f"Inserting into '{TABLE_NAME}' table on Supabase...")
insert_df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="replace",  # Use "append" if you want to stack instead
    index=False,
    dtype={"stats": JSON()}
)
print("✅ Successfully inserted final position averages into Supabase!")
