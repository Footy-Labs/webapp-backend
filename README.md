# FootyLabs - Webapp Backend (Python ETL)

This repository contains the Python scripts responsible for processing Wyscout player data (from Excel files) and populating the FootyLabs Supabase database.

## Purpose

The primary script, `populate_players_table.py`, performs the following steps:
1.  Loads raw player statistics from `.xlsx` files located in the `excel_sheets_players/` directory.
2.  Cleans and processes the data using utility functions from the `wyscout_utils/` directory (calculating derived metrics, simplifying positions, calculating percentiles).
3.  Connects to the configured Supabase PostgreSQL database.
4.  Inserts the Dataframe into the players DB table. The script uses `if_exists='append'`, meaning it adds new rows without deleting existing ones, preserving historical data.

## Prerequisites

Before you begin, ensure you have the following installed:
*   **Python** (Version 3.10 or higher recommended)
*   **pip** (Python package installer, usually comes with Python)
*   **Git** (For cloning the repository)

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual clone URL
    cd webapp-backend
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies for this project.
    ```bash
    # On macOS/Linux
    python3 -m venv .venv

    # On Windows
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    ```bash
    # On macOS/Linux
    source .venv/bin/activate

    # On Windows (Git Bash)
    source .venv/Scripts/activate

    # On Windows (Command Prompt/PowerShell)
    .\.venv\Scripts\activate
    ```
    *(Your terminal prompt should now indicate that you are in the `.venv` environment)*

4.  **Install Dependencies:**
    Install all the required Python packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Database Credentials:**
    This script needs credentials to connect to the Supabase database. The recommended way is using environment variables.

    *   Create a file named `.env` in the root directory of the project (`webapp-backend/`).
    *   Add the following lines to the `.env` file, replacing the placeholders with the actual Supabase database credentials:
        ```dotenv
        SUPABASE_DB_USER="postgres.jbqljjyctbsyawijlxfa"
        SUPABASE_DB_PASSWORD="YOUR_DB_PASSWORD" # Replace with the actual password
        SUPABASE_DB_HOST="aws-0-us-east-1.pooler.supabase.com" 
        SUPABASE_DB_NAME="postgres"
        SUPABASE_DB_PORT=5432
        ```
    *   **IMPORTANT:** Ensure the `.env` file is listed in your `.gitignore` file (it should be by default) so you **never commit your credentials** to Git.

2.  **Excel Data Files:**
    Place the downloaded Wyscout player data Excel files (`.xlsx` format) inside the `excel_sheets_players/` directory. The `load_and_concatenate_player_excels` function will automatically find and process all `.xlsx` files within this folder.
