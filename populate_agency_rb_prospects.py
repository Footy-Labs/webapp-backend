import pandas as pd
from sqlalchemy import create_engine, types
from sqlalchemy.engine import URL # <-- Ensure this is imported
import os
from dotenv import load_dotenv
import re

load_dotenv()

# --- Use the same connection logic as your working script ---
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD") # Raw password from .env
SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME")

# Create the connection URL object
url_object = URL.create(
    "postgresql+psycopg2",
    username=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASSWORD, # Pass the raw password here
    host=SUPABASE_DB_HOST,
    port=5432, # Default PostgreSQL port for Supabase
    database=SUPABASE_DB_NAME
)
# Create the engine using the URL object
engine = create_engine(url_object)
# --- End of consistent connection logic ---


TABLE_NAME = "agency_rb_prospects"
EXCEL_FILE_PATH = "Potential RB Players.xlsx"

EXCEL_TO_DB_MAP = {
    "Player": "player_name",
    "Transfermarkt": "transfermarkt_url",
    "Instagram": "instagram_url",
    "Reached out on": "reached_out_on",
    "Their reponse": "their_response",
    "Goals": "goals",
    "Assists": "assists",
    "Matches played": "matches_played",
    "League": "original_league_name",
    "Team": "original_team_name",
    "Position": "position_excel",
    "Age": "age",
    "Market value": "market_value",
    "Contract expires": "contract_expires",
    "Passport country": "passport_country",
    "Foot": "foot",
    "Height": "height",
    "Weight": "weight",
    "On loan": "on_loan",
    "Successful defensive actions per 90": "successful_defensive_actions_p90",
    "Defensive duels won, %": "defensive_duels_won_pct",
    "Accurate crosses, %": "accurate_crosses_pct",
    "Accurate passes, %": "accurate_passes_pct",
    "Key passes per 90": "key_passes_p90",
    "xA per 90": "xa_p90",
    "Footy Labs Score": "footy_labs_score"
}

DB_COLUMN_TARGET_TYPES = {
    "player_name": "text", "transfermarkt_url": "text", "instagram_url": "text",
    "reached_out_on": "date", "their_response": "text", "goals": "int",
    "assists": "int", "matches_played": "int", "original_league_name": "text",
    "original_team_name": "text", "position_excel": "text", "age": "int",
    "market_value": "int", "contract_expires": "date", "passport_country": "text",
    "foot": "text", "height": "int", "weight": "int", "on_loan": "text",
    "successful_defensive_actions_p90": "numeric",
    "defensive_duels_won_pct": "numeric",
    "accurate_crosses_pct": "numeric",
    "accurate_passes_pct": "numeric",
    "key_passes_p90": "numeric",
    "xa_p90": "numeric",
    "footy_labs_score": "numeric"
}

def clean_value(value, target_type, db_column_name_for_context):
    if pd.isna(value) or str(value).strip() == '':
        return None

    if target_type == "int":
        try:
            if db_column_name_for_context in ["height", "weight"] and (value == 0 or str(value) == '0'):
                return None
            if db_column_name_for_context == "market_value":
                 s_value = str(value)
                 cleaned = re.sub(r'[^\d]', '', s_value)
                 return int(cleaned) if cleaned else None
            return int(float(str(value).replace(',', '')))
        except (ValueError, TypeError):
            print(f"Warning: Could not convert '{value}' to int for column '{db_column_name_for_context}'. Setting to None.")
            return None
    elif target_type == "numeric" or target_type == "float":
        try:
            return float(str(value).replace(',', '.'))
        except (ValueError, TypeError):
            print(f"Warning: Could not convert '{value}' to float for column '{db_column_name_for_context}'. Setting to None.")
            return None
    elif target_type == "date":
        try:
            dt_obj = pd.to_datetime(value, errors='coerce')
            if pd.isna(dt_obj):
                # print(f"Warning: Could not parse date '{value}' for column '{db_column_name_for_context}'. Setting to None.") # Already verbose
                return None
            return dt_obj.strftime('%Y-%m-%d')
        except Exception as e:
            # print(f"Warning: Error processing date '{value}' for column '{db_column_name_for_context}': {e}. Setting to None.") # Already verbose
            return None
    elif target_type == "text":
        return str(value).strip()
    return str(value).strip() if pd.notna(value) else None

def main():
    print(f"Attempting to connect to database: {SUPABASE_DB_HOST} / {SUPABASE_DB_NAME}")
    try:
        with engine.connect() as connection:
            print("✅ Successfully connected to the database.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please check your .env file and Supabase credentials.")
        return

    print(f"\nLoading Excel data from: {EXCEL_FILE_PATH}")
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"❌ ERROR: Excel file not found at '{os.path.abspath(EXCEL_FILE_PATH)}'")
        return

    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
        print(f"Original Excel columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"❌ ERROR: Could not read Excel file: {e}")
        return

    print(f"Loaded {df.shape[0]} rows from Excel.")

    processed_rows_for_db = []
    for index, excel_row in df.iterrows():
        player_name_excel_val = excel_row.get("Player")
        if pd.isna(player_name_excel_val) or str(player_name_excel_val).strip() == "":
            # print(f"Skipping Excel row {index+2} due to missing 'Player' name.") # Already verbose
            continue

        db_row = {}
        for excel_header, db_col_name in EXCEL_TO_DB_MAP.items():
            target_type = DB_COLUMN_TARGET_TYPES.get(db_col_name, "text")
            raw_excel_value = excel_row.get(excel_header)
            cleaned = clean_value(raw_excel_value, target_type, db_col_name)
            db_row[db_col_name] = cleaned
        processed_rows_for_db.append(db_row)

    insert_df = pd.DataFrame(processed_rows_for_db)

    for db_col_name in DB_COLUMN_TARGET_TYPES.keys():
        if db_col_name not in insert_df.columns:
            insert_df[db_col_name] = None
            # print(f"Note: Column '{db_col_name}' was not found/mapped, adding as NULL.") # Already verbose

    sqlalchemy_dtype_mapping = {
        "reached_out_on": types.Date(),
        "contract_expires": types.Date(),
        "goals": types.Integer(),
        "assists": types.Integer(),
        "matches_played": types.Integer(),
        "age": types.Integer(),
        "market_value": types.Integer(),
        "height": types.Integer(),
        "weight": types.Integer(),
        "successful_defensive_actions_p90": types.Numeric(),
        "defensive_duels_won_pct": types.Numeric(),
        "accurate_crosses_pct": types.Numeric(),
        "accurate_passes_pct": types.Numeric(),
        "key_passes_p90": types.Numeric(),
        "xa_p90": types.Numeric(),
        "footy_labs_score": types.Numeric()
    }

    print(f"Prepared {len(insert_df)} valid rows for insertion into '{TABLE_NAME}'.")

    if not insert_df.empty:
        try:
            insert_df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False, dtype=sqlalchemy_dtype_mapping)
            print(f"✅ Successfully populated '{TABLE_NAME}' table.")
        except Exception as e:
            print(f"❌ Error inserting data into '{TABLE_NAME}': {e}")
            print("Sample of data that was attempted (first 5 rows):")
            print(insert_df.head())
            print("\nDataFrame columns sent to to_sql:", insert_df.columns.tolist())
            print("\nDataFrame dtypes:", insert_df.dtypes)
    else:
        print("No valid data to insert after processing.")

if __name__ == "__main__":
    main()