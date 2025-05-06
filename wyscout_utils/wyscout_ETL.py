import pandas as pd
import os

# Function to load and concatenate all Excel files in a given directory
def load_and_concatenate_player_excels(directory_path):
    # List all Excel files in the directory
    excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]

    # Initialize an empty list to store DataFrames
    df_list = []

    # Loop through each Excel file and load it into a DataFrame
    for excel_file in excel_files:
        file_path = os.path.join(directory_path, excel_file)
        df = pd.read_excel(file_path)
        df_list.append(df)

    # Concatenate all DataFrames vertically
    concatenated_df = pd.concat(df_list, ignore_index=True)

    concatenated_df = concatenated_df.drop_duplicates(subset=['Player', 'Team'], keep='first')

    concatenated_df = concatenated_df.reset_index(drop=True)

    return concatenated_df

# Function to load, clean, and concatenate all Excel files in a directory
def load_and_concatenate_games_excels(directory_path):
    # List all Excel files in the directory
    excel_files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]

    # Initialise an empty list to store cleaned data
    df_list = []

    # Loop through each Excel file and load it into a DataFrame
    for excel_file in excel_files:
        file_path = os.path.join(directory_path, excel_file)
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Clean the data by removing the first two rows
        df_cleaned = df.iloc[2:]
        print("file path is {}, column length is {}".format(file_path, len(df_cleaned.columns)))

        # Add the cleaned data to the list
        df_list.append(df_cleaned)

    # Concatenate all DataFrames vertically
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # New column names
    column_names = [
        'Date',
        'Match',
        'Competition',
        'Duration',
        'Team',
        'Scheme',
        'Goals',
        'xG',
        'Total Shots',
        'Shots on Target',
        'Shot Accuracy',  # Calculated as shots on target / total shots
        'Total Passes',
        'Accurate Passes',
        'Pass Accuracy',  # Calculated as accurate passes / total passes
        'Possession %',
        'Total Losses',
        'Low Losses',
        'Medium Losses',
        'High Losses',
        'Total Recoveries',
        'Low Recoveries',
        'Medium Recoveries',
        'High Recoveries',
        'Total Duels',
        'Duels Won',
        'Duels Success %',  # Calculated as duels won / total duels
        'Shots Outside Box',
        'Shots Outside Box on Target',
        'Shots Outside Box Accuracy',  # Calculated as shots outside box on target / total shots outside box
        'Positional Attacks',
        'Positional Attacks with Shots',
        'Positional Attacks Success %',  # Calculated as positional attacks with shots / total positional attacks
        'Counterattacks',
        'Counterattacks with Shots',
        'Counterattack Success %',  # Calculated as counterattacks with shots / total counterattacks
        'Set Pieces',
        'Set Pieces with Shots',
        'Set Piece Success %',  # Calculated as set pieces with shots / total set pieces
        'Corners',
        'Corners with Shots',
        'Corner Success %',  # Calculated as corners with shots / total corners
        'Free Kicks',
        'Free Kicks with Shots',
        'Free Kick Success %',  # Calculated as free kicks with shots / total free kicks
        'Penalties',
        'Penalties Converted',
        'Penalty Success %',  # Calculated as penalties converted / total penalties
        'Total Crosses',
        'Accurate Crosses',
        'Cross Accuracy',  # Calculated as accurate crosses / total crosses
        'Deep Completed Crosses',
        'Deep Completed Passes',
        'Total Penalty Area Entries',
        'Penalty Area Entries (Runs)',
        'Penalty Area Entries (Crosses)',
        'Touches in Penalty Area',
        'Offensive Duels',
        'Offensive Duels Won',
        'Offensive Duels Success %',  # Calculated as offensive duels won / total offensive duels
        'Offsides',
        'Conceded Goals',
        'Shots Against',
        'Shots Against on Target',
        'Shots Against Accuracy',  # Calculated as shots against on target / total shots against
        'Defensive Duels',
        'Defensive Duels Won',
        'Defensive Duels Success %',  # Calculated as defensive duels won / total defensive duels
        'Aerial Duels',
        'Aerial Duels Won',
        'Aerial Duels Success %',  # Calculated as aerial duels won / total aerial duels
        'Sliding Tackles',
        'Successful Sliding Tackles',
        'Sliding Tackle Success %',  # Calculated as successful sliding tackles / total sliding tackles
        'Interceptions',
        'Clearances',
        'Fouls',
        'Yellow Cards',
        'Red Cards',
        'Forward Passes',
        'Accurate Forward Passes',
        'Forward Pass Accuracy',  # Calculated as accurate forward passes / total forward passes
        'Back Passes',
        'Accurate Back Passes',
        'Back Pass Accuracy',  # Calculated as accurate back passes / total back passes
        'Lateral Passes',
        'Accurate Lateral Passes',
        'Lateral Pass Accuracy',  # Calculated as accurate lateral passes / total lateral passes
        'Long Passes',
        'Accurate Long Passes',
        'Long Pass Accuracy',  # Calculated as accurate long passes / total long passes
        'Passes to Final Third',
        'Accurate Passes to Final Third',
        'Pass to Final Third Accuracy',  # Calculated as accurate passes to final third / total passes to final third
        'Progressive Passes',
        'Accurate Progressive Passes',
        'Progressive Pass Accuracy',  # Calculated as accurate progressive passes / total progressive passes
        'Smart Passes',
        'Accurate Smart Passes',
        'Smart Pass Accuracy',  # Calculated as accurate smart passes / total smart passes
        'Throw Ins',
        'Accurate Throw Ins',
        'Throw In Accuracy',  # Calculated as accurate throw ins / total throw ins
        'Goal Kicks',
        'Match Tempo',  # Number of team passes per minute of pure ball possession.
        'Average Passes per Possession',
        'Long Pass %',
        'Average Shot Distance',
        'Average Pass Length',
        'PPDA'
    ]

    # Apply the renaming
    concatenated_df.columns = column_names

    # Convert Date to datetime if it's not already
    concatenated_df['Date'] = pd.to_datetime(concatenated_df['Date'], errors='coerce')

    # Compute additional columns
    concatenated_df['Penalty Area Entries (Passes)'] = concatenated_df['Total Penalty Area Entries'] - (
        concatenated_df['Penalty Area Entries (Runs)'] + concatenated_df['Penalty Area Entries (Crosses)']
    )

    # Compute points earned based on 'Goals' and 'Conceded Goals'
    def calculate_points(row):
        if row['Goals'] > row['Conceded Goals']:
            return 3
        elif row['Goals'] == row['Conceded Goals']:
            return 1
        else:
            return 0

    concatenated_df['Points Earned'] = concatenated_df.apply(calculate_points, axis=1)

    # Drop duplicates and reset index before assigning match day
    concatenated_df = concatenated_df.drop_duplicates().reset_index(drop=True)

    # Function to sort each team's data in descending order (latest match first)
    # and assign "Match Day" numbers in reverse order (highest number = latest match)
    def assign_match_day(group):
        group = group.sort_values(by='Date', ascending=False)
        n = len(group)
        group['Match Day'] = list(range(n, 0, -1))
        return group

    # Apply the grouping by 'Team', then reset the overall index after assigning match days
    concatenated_df = concatenated_df.groupby('Team', group_keys=False).apply(assign_match_day)
    concatenated_df = concatenated_df.reset_index(drop=True)

    return concatenated_df

def process_columns(full_data, cleaned_wyscout_mapping):
    """
    Processes columns by computing derived metrics and updating cleaned_wyscout_mapping.
    Includes calculations for Penalties per 90, Successful Dribbles per 90, Accurate Crosses,
    Accurate Progressive Passes, Accurate Passes to Penalty Area, Accurate Smart Passes,
    Accurate Passes to Final Third, and Shots on Target per 90.
    """

    # Compute Penalties per 90
    if 'Penalties taken' in full_data.columns and 'Minutes played' in full_data.columns:
        full_data['Penalties per 90'] = (full_data['Penalties taken'] / full_data['Minutes played']) * 90
        full_data = full_data.drop(columns=['Penalties taken'])
        cleaned_wyscout_mapping['Set Pieces'] = [
            'Free kicks per 90', 'Direct free kicks per 90',
            'Direct free kicks on target, %', 'Corners per 90',
            'Penalty conversion, %', 'Penalties per 90'
        ]

    # Compute Successful Dribbles per 90
    if 'Dribbles per 90' in full_data.columns and 'Successful dribbles, %' in full_data.columns:
        full_data['Successful dribbles per 90'] = (full_data['Dribbles per 90'] * full_data['Successful dribbles, %']) / 100
        full_data = full_data.drop(columns=['Dribbles per 90', 'Successful dribbles, %'])
        cleaned_wyscout_mapping['Attacking'] = [
            col for col in cleaned_wyscout_mapping['Attacking']
            if col not in ['Dribbles per 90', 'Successful dribbles, %']
        ]
        cleaned_wyscout_mapping['Attacking'].append('Successful dribbles per 90')

    # Compute Accurate Crosses from Left Flank per 90
    if 'Crosses from left flank per 90' in full_data.columns and 'Accurate crosses from left flank, %' in full_data.columns:
        full_data['Accurate crosses from left flank per 90'] = (full_data['Crosses from left flank per 90'] *
                                                               full_data['Accurate crosses from left flank, %']) / 100
        full_data = full_data.drop(columns=['Crosses from left flank per 90', 'Accurate crosses from left flank, %'])
        cleaned_wyscout_mapping['Attacking'] = [
            col for col in cleaned_wyscout_mapping['Attacking']
            if col not in ['Crosses from left flank per 90', 'Accurate crosses from left flank, %']
        ]
        cleaned_wyscout_mapping['Attacking'].append('Accurate crosses from left flank per 90')

    # Compute Accurate Crosses from Right Flank per 90
    if 'Crosses from right flank per 90' in full_data.columns and 'Accurate crosses from right flank, %' in full_data.columns:
        full_data['Accurate crosses from right flank per 90'] = (full_data['Crosses from right flank per 90'] *
                                                                full_data['Accurate crosses from right flank, %']) / 100
        full_data = full_data.drop(columns=['Crosses from right flank per 90', 'Accurate crosses from right flank, %'])
        cleaned_wyscout_mapping['Attacking'] = [
            col for col in cleaned_wyscout_mapping['Attacking']
            if col not in ['Crosses from right flank per 90', 'Accurate crosses from right flank, %']
        ]
        cleaned_wyscout_mapping['Attacking'].append('Accurate crosses from right flank per 90')

    # Compute Accurate Progressive Passes per 90
    if 'Progressive passes per 90' in full_data.columns and 'Accurate progressive passes, %' in full_data.columns:
        full_data['Accurate progressive passes per 90'] = (full_data['Progressive passes per 90'] *
                                                           full_data['Accurate progressive passes, %']) / 100
        full_data = full_data.drop(columns=['Progressive passes per 90', 'Accurate progressive passes, %'])
        cleaned_wyscout_mapping['Key Passing'] = [
            col for col in cleaned_wyscout_mapping['Key Passing']
            if col not in ['Progressive passes per 90', 'Accurate progressive passes, %']
        ]
        cleaned_wyscout_mapping['Key Passing'].append('Accurate progressive passes per 90')

    # Compute Accurate Passes to Penalty Area per 90
    if 'Passes to penalty area per 90' in full_data.columns and 'Accurate passes to penalty area, %' in full_data.columns:
        full_data['Accurate passes to penalty area per 90'] = (full_data['Passes to penalty area per 90'] *
                                                               full_data['Accurate passes to penalty area, %']) / 100
        full_data = full_data.drop(columns=['Passes to penalty area per 90', 'Accurate passes to penalty area, %'])
        cleaned_wyscout_mapping['Key Passing'] = [
            col for col in cleaned_wyscout_mapping['Key Passing']
            if col not in ['Passes to penalty area per 90', 'Accurate passes to penalty area, %']
        ]
        cleaned_wyscout_mapping['Key Passing'].append('Accurate passes to penalty area per 90')

    # Compute Accurate Smart Passes per 90
    if 'Smart passes per 90' in full_data.columns and 'Accurate smart passes, %' in full_data.columns:
        full_data['Accurate smart passes per 90'] = (full_data['Smart passes per 90'] *
                                                     full_data['Accurate smart passes, %']) / 100
        full_data = full_data.drop(columns=['Smart passes per 90', 'Accurate smart passes, %'])
        cleaned_wyscout_mapping['Key Passing'] = [
            col for col in cleaned_wyscout_mapping['Key Passing']
            if col not in ['Smart passes per 90', 'Accurate smart passes, %']
        ]
        cleaned_wyscout_mapping['Key Passing'].append('Accurate smart passes per 90')

    # Compute Accurate Passes to Final Third per 90
    if 'Passes to final third per 90' in full_data.columns and 'Accurate passes to final third, %' in full_data.columns:
        full_data['Accurate passes to final third per 90'] = (full_data['Passes to final third per 90'] *
                                                              full_data['Accurate passes to final third, %']) / 100
        full_data = full_data.drop(columns=['Passes to final third per 90', 'Accurate passes to final third, %'])
        cleaned_wyscout_mapping['Key Passing'] = [
            col for col in cleaned_wyscout_mapping['Key Passing']
            if col not in ['Passes to final third per 90', 'Accurate passes to final third, %']
        ]
        cleaned_wyscout_mapping['Key Passing'].append('Accurate passes to final third per 90')

    # Compute Shots on Target per 90
    if 'Shots per 90' in full_data.columns and 'Shots on target, %' in full_data.columns:
        full_data['Shots on target per 90'] = (full_data['Shots per 90'] * full_data['Shots on target, %']) / 100
        full_data = full_data.drop(columns=['Shots on target, %'])
        cleaned_wyscout_mapping['Attacking'] = [
            col for col in cleaned_wyscout_mapping['Attacking']
            if col not in ['Shots on target, %']
        ]
        cleaned_wyscout_mapping['Attacking'].append('Shots on target per 90')


    # Compute Successful Duels per 90
    if 'Duels per 90' in full_data.columns and 'Duels won, %' in full_data.columns:
        full_data['Duels won per 90'] = (full_data['Duels per 90'] * full_data['Duels won, %']) / 100
        full_data = full_data.drop(columns=['Duels per 90'])
        cleaned_wyscout_mapping['Defensive'] = [
            col for col in cleaned_wyscout_mapping['Defensive']
            if col not in ['Duels per 90']
        ]
        cleaned_wyscout_mapping['Defensive'].append('Duels won per 90')

    # Compute Assists per 90 - xA per 90
    full_data['Assist Overperformance'] = full_data['Assists per 90'] - full_data['xA per 90']

    return full_data


def map_simplified_position(full_data):
    """
    Map each player's position to a simplified position based on the majority rule.
    """
    # Define the position mapping
    position_mapping = {
        'Goalkeeper': ['GK'],
        'Centre Back': ['CB', 'RCB', 'LCB', 'RCB3', 'LCB3'],
        'Full Back': ['RB', 'LB', 'RWB', 'LWB', 'RB5', 'LB5'],
        'Defensive Midfielder': ['DMF', 'LDMF', 'RDMF'],
        'Central Midfielder': ['CMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3'],
        'Attacking Midfielder': ['AMF', 'LAMF', 'RAMF'],
        'Winger': ['LW', 'RW'],
        'Centre Forward': ['CF', 'LWF', 'RWF']
    }

    # Function to map positions
    def map_position(position):
        if not isinstance(position, str):
            return 'Other'

        individual_positions = [pos.strip() for pos in position.split(',')]
        mapped_positions = []
        for pos in individual_positions:
            for category, pos_list in position_mapping.items():
                if pos in pos_list:
                    mapped_positions.append(category)

        if mapped_positions:
            position_counts = pd.Series(mapped_positions).value_counts()
            return position_counts.idxmax()
        else:
            return 'Other'

    # Apply the function to create a new 'Simplified Position' column
    full_data['Simplified Position'] = full_data['Position'].apply(map_position)

    return full_data


