from scipy import stats
from mplsoccer import PyPizza
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import percentileofscore
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import math
from matplotlib.ticker import MaxNLocator



def produce_radar_plot(player_name, df, key_metrics_dict, filter_by_position=False, min_minutes=100):
    """
    Produce a radar plot for a player compared to others in the same position.

    Parameters:
        player_name (str): The name of the player to analyze.
        df (DataFrame): The DataFrame containing player statistics.
        key_metrics_dict (dict): A dictionary containing key metrics for each position,
                                 where keys are positions and values are dictionaries
                                 of (metric name -> (category, color)).
    """
    # Extract the player's position from the 'Simplified Position' column
    player_position = df[df['Player'] == player_name]['Simplified Position']

    # Check if the player is found in the dataset
    if player_position.empty:
        print(f"Player {player_name} not found in the dataset.")
        return

    # Extract the position value
    player_position = player_position.iloc[0]

    # Check if the player's position has a corresponding key metrics dictionary
    if player_position not in key_metrics_dict:
        print(f"No key metrics defined for position: {player_position}")
        return

    key_metrics = key_metrics_dict[player_position]

    # Filter the data by the player's "Simplified Position" if needed
    if filter_by_position:
        df_position = df[(df['Simplified Position'] == player_position)]
    else:
        df_position = df.copy()

    # Extract metric names, categories, and colors from the key_metrics dictionary
    metrics = list(key_metrics.keys())
    categories = [key_metrics[metric][0] for metric in metrics]
    slice_colors = [key_metrics[metric][1] for metric in metrics]

    # Step 1: Filter players who have played more than min_minutes
    df_position = df_position[df_position['Minutes played'] >= min_minutes]

    # Step 2: Remove players with NaN values in any of the key metrics
    df_position = df_position.dropna(subset=metrics)

    # Filter the data for the selected player
    player_details = df_position[df_position['Player'] == player_name][metrics]

    if player_details.empty:
        print(f"Player {player_name} not found in the specified position.")
        return

    # Step 3: Calculate the values and percentiles for the radar plot
    values = [round(player_details[column].iloc[0], 2) for column in metrics]
    percentiles = [int(stats.percentileofscore(df_position[column], player_details[column].iloc[0])) for column in metrics]

    # Print the raw values and percentiles for each metric
    print(f"{player_name} - Raw Values and Percentiles:")
    for metric, value, percentile in zip(metrics, values, percentiles):
        print(f"{metric}: Raw Value = {value}, Percentile = {percentile}%")

    # Step 4: Create the radar plot
    baker = PyPizza(
        params=metrics,
        min_range=None,
        max_range=None,
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-.",
    )

    # Create the pizza plot (this will automatically create the figure)
    fig, ax = baker.make_pizza(
        percentiles,
        figsize=(10, 10),  # Adjust figure size as needed
        param_location=110,
        slice_colors=slice_colors,
        value_colors=["white"] * len(metrics),
        value_bck_colors=slice_colors,
        kwargs_slices=dict(
            facecolor="cornflowerblue", edgecolor="#000000",
            zorder=2, linewidth=1
        ),
        kwargs_params=dict(
            color="#000000", fontsize=12, va="center"
        ),
        kwargs_values=dict(
            color="#000000", fontsize=12, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )
    )

    # Adjust the text to show actual values
    texts = baker.get_value_texts()
    for i, text in enumerate(texts):
        text.set_text(str(values[i]))

    # Step 5: Add title and subtitle
    fig.text(
        0.515, 0.97, f"{player_name} - Key Metrics per 90 minutes", size=18,
        ha="center", color="#000000"
    )
    fig.text(
        0.515, 0.942,
        f"Comparison against {len(df_position)} players in {player_position} | Data Source: Wyscout",
        size=15,
        ha="center", color="#000000"
    )

    # Step 6: Add a legend explaining the color categories
    legend_labels = {category: color for category, color in zip(categories, slice_colors)}
    handles = [plt.Line2D([0], [0], color=color, lw=4) for category, color in legend_labels.items()]
    labels = list(legend_labels.keys())
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Categories")

    # Step 7: Save the figure with a high DPI, including the player's name in the file name
    sanitized_player_name = player_name.replace(" ", "_")
    output_file = f"radar_plot_{sanitized_player_name}_high_dpi.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Optionally, you can display the plot with plt.show()
    plt.show()

def plot_one_player_metric_scatterplot(full_data, player_name, column_name, min_minutes=100, filter_by_position=False,
                           custom_x_label=None, custom_title=None, player_fontsize = 12):
    """
    Plots a comparison of the specified player in the given column
    (e.g., 'Duels won, %') against all other players, while filtering
    out players with less than a specified number of minutes played, except
    for the selected player. Also prints the player's stat value and percentile.

    Optionally, the dataframe can be filtered to only include players with the same
    "Simplified Position" as the input player. Users can also customize the x-axis label
    and title of the plot.

    Args:
    - full_data: pandas DataFrame containing the player data.
    - player_name: Name of the player to highlight.
    - column_name: The column for which the comparison is made (e.g., 'Goals').
    - min_minutes: Minimum number of minutes played to filter the players.
    - filter_by_position: If True, only include players with the same "Simplified Position" as the input player.
    - custom_x_label: Custom text for the x-axis. If None, use the default column name.
    - custom_title: Custom text for the plot title. If None, use a default title with the number of players and min minutes.
    """

    # Ensure the necessary columns exist in the DataFrame
    if column_name not in full_data.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

    if 'Minutes played' not in full_data.columns:
        raise ValueError(f"The column 'Minutes played' does not exist in the DataFrame.")

    if 'Simplified Position' not in full_data.columns:
        raise ValueError(f"The column 'Simplified Position' does not exist in the DataFrame.")

    # Filter the data by the player's "Simplified Position" if needed
    if filter_by_position:
        player_position = full_data[full_data['Player'] == player_name]['Simplified Position'].values[0]
        filtered_data = full_data[(full_data['Simplified Position'] == player_position)]
    else:
        filtered_data = full_data.copy()

    # Filter players with >= min_minutes, but include the selected player regardless of minutes
    filtered_data = filtered_data[(filtered_data['Minutes played'] >= min_minutes) |
                                  (filtered_data['Player'] == player_name)]

    # Check if the resulting DataFrame is empty
    if filtered_data.empty:
        raise ValueError("No data available after applying the filters. Check your DataFrame or filters.")

    # Drop rows with NaN or Inf in the selected column
    filtered_data = filtered_data[filtered_data[column_name].notna()]
    filtered_data = filtered_data[np.isfinite(filtered_data[column_name])]

    # Check again if the resulting DataFrame is empty
    if filtered_data.empty:
        raise ValueError(f"All data is NaN or Inf for the column '{column_name}' after filtering.")

    # Extract the column values for all filtered players
    column_values = filtered_data[column_name].values
    players = filtered_data['Player']

    # Calculate the percentiles
    percentiles = np.percentile(column_values, [40, 80])

    # Assign colors with transparency set to 0.45
    def assign_color(value):
        if value <= percentiles[0]:  # Bottom 40%
            return (0.8, 0, 0, 0.45)  # Dark red with more transparency
        elif percentiles[0] < value <= percentiles[1]:  # 40-80%
            return (0.8, 0.8, 0, 0.45)  # Dark yellow with more transparency
        else:  # Top 20%
            return (0, 0.6, 0, 0.45)  # Dark green with more transparency

    colors = [assign_color(value) for value in column_values]

    # Get the value of the selected player in the specified column
    player_value = filtered_data[filtered_data['Player'] == player_name][column_name].values[0]

    # Calculate the player's percentile using scipy's percentileofscore
    player_percentile = percentileofscore(column_values, player_value, kind='rank')

    # Create the scatter plot with jitter (random noise on y-axis)
    plt.figure(figsize=(12, 8), dpi=300)  # Set a larger figure size

    # Adding random noise (jitter) to the y-axis
    y_jitter = np.random.normal(0, 0.02, size=len(column_values))

    # Plot all players with jitter and color coding
    for value, y, color in zip(column_values, y_jitter, colors):
        plt.scatter(value, y, color=color, s=100, zorder=3)

    # Highlight the selected player with the new color #31348D
    selected_color = '#31348D'
    plt.scatter(player_value, 0, color=selected_color, s=200, zorder=5)  # Larger point for the selected player

    # Add text annotation with smaller distance from the selected player's dot
    plt.text(player_value, 0.01, player_name, color='black', fontsize=player_fontsize, weight='bold',
             ha='center')  # Reduced distance

    # Add a horizontal line in the middle of the y-axis
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Horizontal line at y=0

    # Calculate x-axis limits with 10% extra on both sides
    value_range = column_values.max() - column_values.min()
    padding = 0.1 * value_range
    x_min = column_values.min() - padding
    x_max = column_values.max() + padding
    plt.xlim(x_min, x_max)

    # Add labels "Low" and "High" in fixed positions relative to the data range
    plt.text(x_min - padding * 0.6, 0, 'Low', color='black', fontsize=12,
             verticalalignment='center')  # Position for "Low"
    plt.text(x_max + padding * 0.2, 0, 'High', color='black', fontsize=12,
             verticalalignment='center')  # Position for "High"

    # Default title and x-axis label
    num_players = len(filtered_data)
    default_title = f'{column_name} - Comparison of {num_players} Players (Min {min_minutes} minutes played this season)'
    default_x_label = f'{column_name}'

    # Set title and x-axis label, using the custom ones if provided
    plt.title(custom_title if custom_title else default_title)
    plt.xlabel(custom_x_label if custom_x_label else default_x_label)

    # Remove y-axis ticks since we don't need them for a horizontal plot
    plt.yticks([])

    # Remove grid lines
    plt.grid(False)

    # Create a custom legend for percentile colours
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.8, 0, 0, 0.45), markersize=10, label='Bottom 40%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.8, 0.8, 0, 0.45), markersize=10, label='40-80%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0.6, 0, 0.45), markersize=10, label='Top 20%')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Print the player's value and percentile
    print(f"{player_name}'s {column_name}: {player_value}")
    print(f"{player_name}'s percentile in {column_name}: {player_percentile:.2f}%")

    # Display the plot
    plt.show()


def compare_2players(full_data, player1, player2, cleaned_wyscout_mapping, top_n=10):
    """
    Compare two players based on percentiles and create two plots:
    1) A plot that highlights their similarities.
    2) A plot that highlights their differences.

    Args:
    - full_data: pandas DataFrame containing player data.
    - player1: Name of the first player to compare.
    - player2: Name of the second player to compare.
    - cleaned_wyscout_mapping: Dictionary with the columns to compare.
    - top_n: Number of most/least similar variables to display (default 10).
    """

    # Step 1: Check if both players exist in the data
    if player1 not in full_data['Player'].values:
        print(f"Error: {player1} not found in the dataset.")
        return
    if player2 not in full_data['Player'].values:
        print(f"Error: {player2} not found in the dataset.")
        return

    # Step 2: Extract the data for both players
    player1_data = full_data[full_data['Player'] == player1].iloc[0]
    player2_data = full_data[full_data['Player'] == player2].iloc[0]

    # Step 3: Define the relevant categories from cleaned_wyscout_mapping
    relevant_categories = ['Performance', 'Defensive', 'Attacking', 'Passing', 'Key Passing', 'Set Pieces']

    # Step 4: Collect all the relevant metrics from these categories
    relevant_columns = sum([cleaned_wyscout_mapping[category] for category in relevant_categories], [])

    # Step 5: Filter the columns that exist in full_data and are numeric
    numeric_columns = full_data.select_dtypes(include=[np.number]).columns
    relevant_numeric_columns = [col for col in relevant_columns if col in numeric_columns]

    # Step 6: Extract the numeric data for the relevant columns for both players
    player1_metrics = player1_data[relevant_numeric_columns]
    player2_metrics = player2_data[relevant_numeric_columns]

    # Step 7: Filter out variables where both players have zero values
    non_zero_metrics = (player1_metrics != 0) | (player2_metrics != 0)
    player1_metrics = player1_metrics[non_zero_metrics]
    player2_metrics = player2_metrics[non_zero_metrics]

    # Step 8: Compute the percentiles for all players in the relevant metrics
    percentiles = full_data[relevant_numeric_columns].rank(pct=True) * 100

    # Get the percentiles for the two players
    player1_percentiles = percentiles.loc[full_data['Player'] == player1].iloc[0]
    player2_percentiles = percentiles.loc[full_data['Player'] == player2].iloc[0]

    # Step 9: Compute the absolute differences between the percentiles of the two players
    percentile_diff = np.abs(player1_percentiles - player2_percentiles)

    # Step 10: Sort by similarity (small differences) and differences (large differences)
    sorted_diff = percentile_diff.sort_values()
    most_similar = sorted_diff.head(top_n).index
    least_similar = sorted_diff.tail(top_n).index

    # Step 11: Plot the most similar metrics
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Create bar width offset to avoid overlapping bars
    bar_width = 0.35
    indices = np.arange(len(most_similar))

    # Plot similar percentiles
    ax[0].barh(indices, player1_percentiles[most_similar], height=bar_width, color='blue', alpha=0.6, label=player1)
    ax[0].barh(indices + bar_width, player2_percentiles[most_similar], height=bar_width, color='green', alpha=0.6,
               label=player2)
    ax[0].set_yticks(indices + bar_width / 2)
    ax[0].set_yticklabels(most_similar)
    ax[0].set_title(f'Top {top_n} Similarities between {player1} and {player2}')
    ax[0].set_xlabel('Percentile Rank (0-100)')
    ax[0].set_xlim(0, 100)
    ax[0].legend()

    # Plot least similar metrics
    indices = np.arange(len(least_similar))
    ax[1].barh(indices, player1_percentiles[least_similar], height=bar_width, color='blue', alpha=0.6, label=player1)
    ax[1].barh(indices + bar_width, player2_percentiles[least_similar], height=bar_width, color='green', alpha=0.6,
               label=player2)
    ax[1].set_yticks(indices + bar_width / 2)
    ax[1].set_yticklabels(least_similar)
    ax[1].set_title(f'Top {top_n} Differences between {player1} and {player2}')
    ax[1].set_xlabel('Percentile Rank (0-100)')
    ax[1].set_xlim(0, 100)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# Define a function to generate the plots for a player based on their Simplified Position
def plot_all_metrics_for_player_scatterplot(metrics_dict, player_name, full_data, min_minutes=100, filter_by_position=False, player_fontsize = 12):
    """
    Loops through all the metrics for the player's position (extracted from the 'Simplified Position')
    and generates plots for each metric. Optionally filters the data by the player's position.

    Args:
    - metrics_dict: Dictionary containing the metrics for each position.
    - player_name: The player to highlight in each plot.
    - full_data: The pandas DataFrame containing the player data.
    - min_minutes: Minimum number of minutes played to filter the players.
    - filter_by_position: If True, only include players with the same 'Simplified Position' as the player.
    """

    # Ensure the player exists in the dataset
    if player_name not in full_data['Player'].values:
        print(f"Player '{player_name}' not found in the dataset.")
        return

    # Extract the player's "Simplified Position"
    player_position = full_data[full_data['Player'] == player_name]['Simplified Position'].values[0]

    # Ensure the player's position exists in the metrics dictionary
    if player_position not in metrics_dict:
        print(f"Position '{player_position}' not found in the metrics dictionary.")
        return

    # Get the relevant metrics for the player's position
    position_metrics = metrics_dict[player_position]

    # Loop through each metric and plot it
    for metric, (category, color) in position_metrics.items():
        print(f"Plotting {metric} for {player_name} in {player_position} category '{category}'...")
        try:
            plot_one_player_metric_scatterplot(full_data, player_name=player_name, column_name=metric,
                                   min_minutes=min_minutes, filter_by_position=filter_by_position,
                                   player_fontsize = player_fontsize)
        except ValueError as e:
            print(f"Error plotting {metric}: {e}")
        print()  # Add a space between plots


def plot_one_player_metric_histogram(full_data, player_name, column_name, min_minutes=100, filter_by_position=False,
                                     custom_x_label=None, custom_title=None, player_fontsize=12):
    """
    Improved plot with shaded PDF, custom fonts, and design enhancements.
    """

    # Ensure the necessary columns exist in the DataFrame
    if column_name not in full_data.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

    if 'Minutes played' not in full_data.columns:
        raise ValueError(f"The column 'Minutes played' does not exist in the DataFrame.")

    if 'Simplified Position' not in full_data.columns:
        raise ValueError(f"The column 'Simplified Position' does not exist in the DataFrame.")

    # Filter the data by the player's "Simplified Position" if needed
    if filter_by_position:
        player_position = full_data[full_data['Player'] == player_name]['Simplified Position'].values[0]
        filtered_data = full_data[(full_data['Simplified Position'] == player_position)]
    else:
        filtered_data = full_data.copy()

    # Filter players with >= min_minutes, but include the selected player regardless of minutes
    filtered_data = filtered_data[(filtered_data['Minutes played'] >= min_minutes) |
                                  (filtered_data['Player'] == player_name)]

    # Drop rows with NaN or Inf in the selected column
    filtered_data = filtered_data[filtered_data[column_name].notna()]
    filtered_data = filtered_data[np.isfinite(filtered_data[column_name])]

    # Extract the column values for all filtered players
    column_values = filtered_data[column_name].values
    player_value = filtered_data[filtered_data['Player'] == player_name][column_name].values[0]

    # Calculate percentiles
    median = np.median(column_values)
    percentile_90 = np.percentile(column_values, 90)  # Calculate 90th percentile
    player_percentile = percentileofscore(column_values, player_value, kind='rank')

    # Create frequency distribution (histogram)
    plt.figure(figsize=(10, 6), dpi=300)
    counts, bins, _ = plt.hist(column_values, bins=30, alpha=0.6, color='#DCC6E0', density=True)  # Pastel violet
    max_hist_y = max(counts)  # Find the maximum y-value of the histogram

    # Overlay the PDF (using blue)
    xmin, xmax = bins.min(), bins.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, column_values.mean(), column_values.std())
    plt.plot(x, p, color='blue', linewidth=2)  # PDF in blue
    plt.fill_between(x, p, color='blue', alpha=0.1)

    # Add vertical line for the median (using red)
    plt.axvline(median, color='red', linestyle='--', linewidth=1.5, label='Median')

    # Add vertical line for the 90th percentile (using green dotted line)
    plt.axvline(percentile_90, color='green', linestyle=':', linewidth=1.5, label='90th Percentile')

    # Highlight the selected player (player's dot in dark blue)
    plt.scatter(player_value, 0, color='#2A9D8F', s=200, zorder=5)  # Player's dot in greenish-blue

    # Set the y-offset proportional to the histogram's max value
    arrow_y_offset = max_hist_y * 0.2  # 20% of the max histogram value for the arrow height

    # Draw the arrow pointing at the player's value
    plt.annotate('', xy=(player_value, 0), xytext=(player_value, arrow_y_offset),
                 arrowprops=dict(facecolor='#2A9D8F', shrink=0.05, width=1.5, headwidth=8))

    # Move player name above the arrow, ensuring enough space regardless of the PDF height
    text_y_position = arrow_y_offset + max_hist_y * 0.03  # Ensure text is always above the arrow
    plt.text(player_value, text_y_position, player_name, color='#264653', fontsize=player_fontsize, weight='bold',
             ha='center')

    # Customize graph
    plt.xlabel(custom_x_label if custom_x_label else column_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(
        custom_title if custom_title else f'{column_name} - Comparison of {len(filtered_data)} Players (Min {min_minutes} minutes played this season)')

    # Show plot
    plt.show()

    # Print player's value and percentile
    print(f"{player_name}'s {column_name}: {player_value}")
    print(f"{player_name}'s percentile in {column_name}: {player_percentile:.2f}%")


def plot_all_metrics_for_player_histogram(metrics_dict, player_name, full_data, min_minutes=100, filter_by_position=False, player_fontsize = 12):
    """
    Loops through all the metrics for the player's position (extracted from the 'Simplified Position')
    and generates plots for each metric. Optionally filters the data by the player's position.

    Args:
    - metrics_dict: Dictionary containing the metrics for each position.
    - player_name: The player to highlight in each plot.
    - full_data: The pandas DataFrame containing the player data.
    - min_minutes: Minimum number of minutes played to filter the players.
    - filter_by_position: If True, only include players with the same 'Simplified Position' as the player.
    """

    # Ensure the player exists in the dataset
    if player_name not in full_data['Player'].values:
        print(f"Player '{player_name}' not found in the dataset.")
        return

    # Extract the player's "Simplified Position"
    player_position = full_data[full_data['Player'] == player_name]['Simplified Position'].values[0]

    # Ensure the player's position exists in the metrics dictionary
    if player_position not in metrics_dict:
        print(f"Position '{player_position}' not found in the metrics dictionary.")
        return

    # Get the relevant metrics for the player's position
    position_metrics = metrics_dict[player_position]

    # Loop through each metric and plot it
    for metric, (category, color) in position_metrics.items():
        print(f"Plotting {metric} for {player_name} in {player_position} category '{category}'...")
        try:
            plot_one_player_metric_histogram(full_data, player_name=player_name, column_name=metric,
                                   min_minutes=min_minutes, filter_by_position=filter_by_position,
                                   player_fontsize = player_fontsize)
        except ValueError as e:
            print(f"Error plotting {metric}: {e}")
        print()  # Add a space between plots



# Function to create a bar chart comparing averages for multiple columns across multiple teams
def plot_multi_team_bar_comparison(df_all, team_names, columns, n_games="all"):
    # Set the plot style for better visualisation
    sns.set(style="whitegrid")

    # Initialise a list to store each team's averages
    averages_list = []

    # Iterate over team names and filter data for each team
    for team_name in team_names:
        # Filter the dataframe for the specified team
        df_filtered = df_all[df_all['Team'] == team_name].copy()

        # Filter to last n_games if specified
        if n_games != "all" and isinstance(n_games, int):
            df_filtered = df_filtered.head(n_games)

        # Compute the averages for the specified columns
        team_averages = df_filtered[columns].mean()

        # Append the team averages to the list with the team name.
        # Using team_averages.values to ensure proper alignment.
        averages_list.append(pd.DataFrame({
            'Metric': columns,
            'Team': team_name,
            'Average': team_averages.values
        }))

    # Concatenate all the team averages into a single dataframe
    avg_df_melted = pd.concat(averages_list, ignore_index=True)

    # Suppress Seaborn's FutureWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Create a figure and axis object
        plt.figure(figsize=(10, 6), dpi=300)

        # Create the bar plot with side-by-side bars for each team
        ax = sns.barplot(x='Team', y='Average', hue='Metric', data=avg_df_melted)

        # Add labels on top of each bar
        # Add labels on top of each bar, but only if the bar's height is nonzero
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.2f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5),
                            textcoords='offset points', fontsize=12)

        # Set plot labels and title
        plt.xlabel("Team")
        plt.ylabel("Average Value")
        if n_games == "all":
            title = f"Average {', '.join(columns)} Comparison Across Teams (Whole Season)"
        else:
            title = f"Average {', '.join(columns)} Comparison Across Teams (Last {n_games} Games)"
        plt.title(title)

        # Display a legend
        plt.legend(title='Metrics')

        # Show the plot
        plt.tight_layout()
        plt.show()


def plot_multi_team_lines_comparison(df_all, team_names, column_name, n_games="all"):
    sns.set(style="whitegrid")
    filtered_list = []
    for team_name in team_names:
        df_filtered = df_all[df_all['Team'] == team_name].copy()
        if n_games != "all" and isinstance(n_games, int):
            df_filtered = df_filtered.head(n_games)
        df_filtered.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        team_average = df_filtered[column_name].mean()
        print(
            f"Average {column_name} for {team_name} over the last {n_games if n_games != 'all' else 'whole season'}: {team_average:.2f}")
        filtered_list.append((df_filtered, team_name))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for df_filtered, team_name in filtered_list:
            sns.lineplot(x='Match Day', y=column_name, data=df_filtered, marker='o', label=team_name, ax=ax)
        if n_games == "all":
            title = f"{column_name} Comparison Across Teams (Whole Season)"
        else:
            title = f"{column_name} Comparison Across Teams (Last {n_games} Games)"
        ax.set_xlabel("Match Day")
        ax.set_ylabel(column_name)
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(title='Teams')
        plt.tight_layout()
        plt.show()


def compare_2teams(dfs, team_names, n_games="all", top_n=10):
    """
    Compare two teams based on percentage differences (in %) over the last n_games and highlight similarities and differences.

    Args:
    - dfs: List of dataframes containing team data.
    - team_names: List of team names corresponding to the dataframes.
    - n_games: Number of last games to include in the comparison (default is "all").
    - top_n: Number of most/least similar metrics to display (default is 10).
    """
    # Step 1: Extract the two teams' data
    df_team1 = dfs[0][dfs[0]['Team'] == team_names[0]].copy()
    print("df_team1 shape is: ", df_team1.shape)
    df_team2 = dfs[1][dfs[1]['Team'] == team_names[1]].copy()
    print("df_team2 shape is: ", df_team2.shape)

    # Step 2: If n_games is specified, filter the last n_games
    if n_games != "all" and isinstance(n_games, int):
        df_team1 = df_team1.head(n_games)
        df_team2 = df_team2.head(n_games)

    # Step 3: Calculate the averages for all numerical columns for both teams, excluding 'Duration'
    team1_averages = df_team1.select_dtypes(include=[np.number]).drop(
        columns=['Duration', 'Penalties Converted', 'Penalties', 'Penalty Success %'], errors='ignore').mean()
    team2_averages = df_team2.select_dtypes(include=[np.number]).drop(
        columns=['Duration', 'Penalties Converted', 'Penalties', 'Penalty Success %'], errors='ignore').mean()

    # Step 4: Calculate the percentage difference and multiply by 100 to express in %
    percentage_diff = (team1_averages - team2_averages).abs() / ((team1_averages + team2_averages) / 2) * 100

    # Step 5: Exclude cases where both teams have 0 averages (to avoid division by zero issues)
    non_zero_metrics = (team1_averages != 0) | (team2_averages != 0)
    percentage_diff = percentage_diff[non_zero_metrics]
    team1_averages = team1_averages[non_zero_metrics]
    team2_averages = team2_averages[non_zero_metrics]

    # Step 6: Sort by similarity (small percentage differences) and differences (large percentage differences)
    sorted_diff = percentage_diff.sort_values()

    # Step 7: Select the top N similarities and differences
    most_similar = sorted_diff.head(top_n).index
    least_similar = sorted_diff.tail(top_n).index

    # Step 8: Plot the most similar and least similar metrics (using percentage differences)
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    # Font sizes
    title_fontsize = 19
    label_fontsize = 18
    tick_fontsize = 16
    bar_label_fontsize = 16

    # Create bar width offset to avoid overlapping bars
    bar_width = 0.35
    indices = np.arange(len(most_similar))

    # Plot similar metrics
    ax[0].barh(indices, percentage_diff[most_similar].iloc[:], height=bar_width, color='blue', alpha=0.6)
    for i, index in enumerate(indices):
        ax[0].text(percentage_diff[most_similar].iloc[i], index,
                   f'{team1_averages[most_similar].iloc[i]:.2f} / {team2_averages[most_similar].iloc[i]:.2f}',
                   va='center', fontsize=bar_label_fontsize)

    ax[0].set_yticks(indices)
    ax[0].set_yticklabels(most_similar, fontsize=tick_fontsize)
    ax[0].set_title(f'Top {top_n} Similarities (Percentage difference close to 0%)', fontsize=title_fontsize)
    ax[0].set_ylabel('Metric', fontsize=label_fontsize)

    # Plot least similar metrics
    indices = np.arange(len(least_similar))
    ax[1].barh(indices, percentage_diff[least_similar].iloc[:], height=bar_width, color='red', alpha=0.6)
    for i, index in enumerate(indices):
        ax[1].text(percentage_diff[least_similar].iloc[i], index,
                   f'{team1_averages[least_similar].iloc[i]:.2f} / {team2_averages[least_similar].iloc[i]:.2f}',
                   va='center', fontsize=bar_label_fontsize)

    ax[1].set_yticks(indices)
    ax[1].set_yticklabels(least_similar, fontsize=tick_fontsize)
    ax[1].set_title(f'Top {top_n} Differences (Percentage difference far from 0%)', fontsize=title_fontsize)

    # Manually add a single x-axis label for the entire figure
    fig.text(0.5, 0.04, f'Percentage Difference (%) ({team_names[0]} / {team_names[1]})', ha='center',
             fontsize=label_fontsize)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the x-axis label
    plt.show()


def plot_goals_xg_comparison(dfs, team_names, n_games="all"):
    """
    Plot the comparison of Goals scored, Goals conceded, xG, and xG against for multiple teams with grouped and color-coded bars.

    Args:
    - dfs: List of dataframes containing team data.
    - team_names: List of team names corresponding to the dataframes.
    - n_games: Number of last games to include in the comparison (default is "all").
    """
    # Initialize a list to store each team's averages
    averages_list = []

    # Iterate through the dataframes and team names
    for df, team_name in zip(dfs, team_names):
        # Filter the dataframe to only include rows for the specified team
        df_filtered = df[df['Team'] == team_name].copy()
        df_opponents = df[df['Team'] != team_name].copy()  # Rows for the opponent teams

        # Filter to last N games if n_games is specified
        if n_games != "all" and isinstance(n_games, int):
            df_filtered = df_filtered.head(n_games)
            df_opponents = df_opponents.head(n_games)

        # Compute Goals and xG stats
        goals_scored = df_filtered['Goals'].mean()  # Goals scored by the team
        goals_conceded = df_opponents['Goals'].mean()  # Goals conceded by the team (goals from opponent rows)
        xg_for = df_filtered['xG'].mean()  # Expected goals by the team
        xg_against = df_opponents['xG'].mean()  # Expected goals conceded (xG from opponent rows)

        # Append the team averages to the list with the team name
        averages_list.append(pd.DataFrame({
            'Metric': ['Goals Scored', 'Expected Goals (xG)', 'Goals Conceded', 'Expected Goals Conceded (xG)'],
            'Team': team_name,
            'Average': [goals_scored, xg_for, goals_conceded, xg_against]
        }))

    # Concatenate all the team averages into a single dataframe
    avg_df_melted = pd.concat(averages_list, ignore_index=True)

    # Define the color palette with lighter and darker shades of green and red
    palette = {
        'Goals Scored': '#0cf014',  # Bright green
        'Expected Goals (xG)': '#73f077',  # Lighter green
        'Goals Conceded': '#e60e0e',  # Dark red
        'Expected Goals Conceded (xG)': '#ed3939'  # Lighter red
    }

    # Sort the data to group metrics properly (so goals and xG are side by side)
    avg_df_melted['Metric'] = pd.Categorical(avg_df_melted['Metric'],
                                             ['Goals Scored', 'Expected Goals (xG)', 'Goals Conceded',
                                              'Expected Goals Conceded (xG)'])

    # Suppress Seaborn's FutureWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Create a figure and axis object
        plt.figure(figsize=(13, 7), dpi=300)

        # Create the bar plot with grouped bars side by side for each team
        ax = sns.barplot(x='Team', y='Average', hue='Metric', data=avg_df_melted, palette=palette, dodge=True)

        # Add labels on top of the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=12)

        # Determine title based on the number of games analyzed
        if n_games == "all":
            title = "Goals and Expected Goals (xG) Comparison Across Teams (Whole Season)"
        else:
            title = f"Goals and Expected Goals (xG) Comparison Across Teams (Last {n_games} Games)"

        # Set plot labels and title
        plt.xlabel("Team", fontsize=14)
        plt.ylabel("Average Value", fontsize=14)
        plt.title(title, fontsize=16)

        # Make the team names on the x-axis larger
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)  # Increase the size of the team names

        # Display a legend
        plt.legend(title='Metrics')

        # Show the plot
        plt.tight_layout()
        plt.show()


def plot_goal_distribution(df, team_name='Džiugas'):
    # Extract the data for the given team
    team_data = df[df['Team'] == team_name]

    # Extract goals scored and conceded data
    time_slots = ['0-15 mn', '16-30 mn', '31-45 mn', '46-60 mn', '61-75 mn', '76-90 mn']
    goals_scored = team_data.iloc[0, 1:7].values  # Goals scored columns
    goals_conceded = team_data.iloc[0, 7:].values  # Goals conceded columns

    # Create the plot
    plt.figure(figsize=(10, 6), dpi=300)

    # Highlight half-time (between 31-45 min and 46-60 min)
    plt.axvspan(2.02, 2.98, color='#FFBF00', alpha=1)  # Black block indicating half-time
    plt.text(2.5, max(goals_scored.max(), goals_conceded.max()) / 2, 'Half-Time', color='white',
             ha='center', va='center', fontsize=12, fontweight='bold')

    # Plot goals scored in green (split into two segments to avoid crossing the black block)
    plt.plot(time_slots[:3], goals_scored[:3], color='green', marker='o', label='Goals Scored', zorder=1)
    plt.plot(time_slots[3:], goals_scored[3:], color='green', marker='o', zorder=1)

    # Plot goals conceded in red (split into two segments to avoid crossing the black block)
    plt.plot(time_slots[:3], goals_conceded[:3], color='red', marker='o', label='Goals Conceded', zorder=1)
    plt.plot(time_slots[3:], goals_conceded[3:], color='red', marker='o', zorder=1)

    # Add labels and title
    plt.xlabel('Time Slot')
    plt.ylabel('Number of Goals')
    plt.title(f'Goal Distribution for {team_name} (2024)')

    # Set y-axis to start from 0
    plt.ylim(0, None)

    # Show legend (only two labels now)
    plt.legend()

    # Display the plot
    plt.show()


def plot_and_calculate_plus_minus(df_goal_distribution, df_total, team_name, min_minutes=1000, top_n=5,
                                  language='english'):
    # Get total goals scored and conceded for the team from df_goal_distribution
    team_data = df_goal_distribution[df_goal_distribution['Team'] == team_name]

    team_goals_scored = team_data.iloc[:, 1:7].sum().sum()  # Sum across all 'Goals Scored' columns
    team_goals_conceded = team_data.iloc[:, 7:].sum().sum()  # Sum across all 'Goals Conceded' columns
    goal_difference = team_goals_scored - team_goals_conceded

    # Filter df_total for the selected team and players who have played at least min_minutes
    filtered_players = df_total[(df_total['team'] == team_name) & (df_total['minutes_played'] >= min_minutes)]

    # Sort players by plus-minus per minute in descending order and select the top_n players
    top_players = filtered_players.nlargest(top_n, 'plus_minus_per_minute')

    # Prepare data for plotting
    players = pd.Series(list(top_players['player']))
    plus_minus_values = pd.Series(list(top_players['plus_minus']))
    minutes_played = list(top_players['minutes_played'])

    # Add team-wide goal difference at the top
    players = pd.concat([pd.Series([f"{team_name} (Overall)"]), players], ignore_index=True)
    plus_minus_values = pd.concat([pd.Series([goal_difference]), plus_minus_values], ignore_index=True)
    minutes_played.insert(0, None)  # No minutes played for the team total

    # Create the color palette: team in blue, players in yellow (#FFBF00), and grey for zero goal difference
    palette = ['blue' if v != 0 else 'grey' for v in plus_minus_values]
    palette[1:] = ['#FFBF00' if v != 0 else 'grey' for v in plus_minus_values[1:]]

    # Create the bar plot with edgecolor
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=players, y=plus_minus_values, palette=palette,
                edgecolor='black')  # Adding a black edge color around each bar

    # Set plot title and labels
    plt.title(f"{team_name} Goal Difference and Top {top_n} Players", fontsize=14)
    plt.xlabel("Player/Team", fontsize=12)
    plt.ylabel("Goal Difference", fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")

    # Set y-axis to display only even ticks and add grid lines
    min_val = int(min(plus_minus_values)) - 1
    max_val = int(max(plus_minus_values)) + 2
    plt.yticks(np.arange(min_val - (min_val % 2), max_val + (max_val % 2), 2))
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Add labels on top of each player's bar (only for players, not for the team total)
    for i, value in enumerate(plus_minus_values):
        if i != 0:  # Skip label for the team total
            plt.text(i, value + 0.1 if value >= 0 else value - 0.1, f'{int(minutes_played[i])} mins', ha='center',
                     va='bottom' if value >= 0 else 'top')

    plt.tight_layout()
    plt.show()

    # Create language-specific text for the top players
    if language == 'english':
        output_text = (
            f"{team_name} has scored {team_goals_scored} goals this season and conceded {team_goals_conceded} goals "
            f"this season. Therefore, its overall goal difference is {goal_difference}.\n"
            f"The top {top_n} players for {team_name}, based on their time on the pitch and goal difference per minute, are:\n")

        for index, row in top_players.iterrows():
            goal_text = "goals" if abs(row['plus_minus']) != 1 else "goal"
            if row['plus_minus'] > 0:
                output_text += (f"{row['player']} - While {row['player']} was on the pitch, the team scored "
                                f"{row['plus_minus']} more {goal_text} than they conceded over {row['minutes_played']} minutes played.\n")
            elif row['plus_minus'] < 0:
                output_text += (f"{row['player']} - While {row['player']} was on the pitch, the team conceded "
                                f"{abs(row['plus_minus'])} more {goal_text} than they scored over {row['minutes_played']} minutes played.\n")
            else:
                output_text += (
                    f"{row['player']} - While {row['player']} was on the pitch, the team’s goal difference remained neutral "
                    f"over {row['minutes_played']} minutes played.\n")

    elif language == 'lithuanian':
        output_text = (f"{team_name} šį sezoną įmušė {team_goals_scored} " +
                       (f"įvartį" if team_goals_scored == 1 else "įvarčius") +
                       f" ir praleido {team_goals_conceded} " +
                       (f"įvartį" if team_goals_conceded == 1 else "įvarčius") +
                       f". Todėl bendras įvarčių skirtumas yra {goal_difference}.\n"
                       f"Top {top_n} žaidėjai pagal įvarčių skirtumą per minutę, kai jie buvo aikštėje, yra:\n")

        for index, row in top_players.iterrows():
            if abs(row['plus_minus']) == 1:
                goal_text = "vienu įvarčiu"
            else:
                goal_text = f"{row['plus_minus']} įvarčiais" if abs(row['plus_minus']) > 1 else "įvarčiais"

            if row['plus_minus'] > 0:
                output_text += (f"{row['player']} - Kai {row['player']} buvo aikštėje, komanda įmušė "
                                f"{goal_text} daugiau nei praleido per {row['minutes_played']} sužaistas minutes.\n")
            elif row['plus_minus'] < 0:
                output_text += (f"{row['player']} - Kai {row['player']} buvo aikštėje, komanda praleido "
                                f"{goal_text} daugiau nei įmušė per {row['minutes_played']} sužaistas minutes.\n")
            else:
                output_text += (
                    f"{row['player']} - Kai {row['player']} buvo aikštėje, komandos įvarčių skirtumas išliko neutralus "
                    f"per {row['minutes_played']} sužaistas minutes.\n")

    print(output_text)


def plot_top_stats_with_minutes(all_league_players_df, team_name, stats_list, top_n=5, min_minutes_players = 500):
    # Set the plot style for better visualization
    sns.set(style="whitegrid")

    # Filter the dataframe for the given team
    team_data = all_league_players_df[all_league_players_df['Team'] == team_name]

    # Filter the dataframe for players who have played at least n_minutes
    team_data = team_data[team_data['Minutes played'] >= min_minutes_players]

    # Determine the number of subplots needed based on the number of stats
    num_stats = len(stats_list)
    cols = 2  # Number of columns in the plot grid (2 per row)
    rows = math.ceil(num_stats / cols)  # Calculate the number of rows needed

    # Create subplots
    fig, ax = plt.subplots(rows, cols, figsize=(14, 6 * rows), sharey=False)  # Each plot will have its own Y-axis scale
    ax = ax.flatten()  # Flatten the axis array for easier indexing if rows > 1

    # Loop through each stat and create its respective bar plot
    for i, stat in enumerate(stats_list):
        # Check if the statistic exists in the dataframe
        if stat not in team_data.columns:
            print(f"Statistic '{stat}' not found in the dataframe. Skipping.")
            continue

        # Get the top n players for the current stat
        top_players = team_data[['Player', stat, 'Minutes played']].nlargest(top_n, stat)

        # Plot vertical bars
        sns.barplot(x='Player', y=stat, data=top_players, ax=ax[i], color='#FFBF00')
        ax[i].set_title(f"Top {top_n} Players for {stat}")
        ax[i].set_xlabel('Player')
        ax[i].set_ylabel(f'{stat}')

        # Add the values (stat and minutes played) on top of the bars
        for index, p in enumerate(ax[i].patches):
            height = p.get_height()
            minutes = top_players.iloc[index]['Minutes played']
            ax[i].annotate(f"{int(minutes)} mn played",
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center', xytext=(0, 9),
                           textcoords='offset points')

    # Remove unused subplots, if any
    for j in range(i + 1, len(ax)):
        fig.delaxes(ax[j])

    # Adjust layout
    plt.suptitle(f"Top {top_n} Players by Stat for {team_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_player_resume_table(df, player_name, additional_info):
    """
    Creates a styled player profile table for the given player, allowing for overwriting of certain metrics while maintaining the original order.

    Parameters:
    - df: DataFrame containing player data.
    - player_name: The name of the player to create the profile for.
    - additional_info: A dictionary containing additional information to be added to or overwrite existing data.

    Returns:
    - A styled DataFrame displaying the player's profile.
    """
    # Filter the DataFrame for the player and select the relevant columns
    player_data = df[df['Player'] == player_name][['Team', 'Simplified Position', 'Age', 'Foot',
        'Height', 'Weight', 'Goals', 'Assists', 'Matches played', 'Minutes played', 'Birth country']]

    # Update the age to be an integer
    player_data['Age'] = player_data['Age'].astype(int)

    # Transpose the DataFrame and reset the index
    player_data_transposed = player_data.T.reset_index()

    # Rename the columns for clarity
    player_data_transposed.columns = ['Metric', 'Value']

    # Overwrite values with the additional_info dictionary
    for key, value in additional_info.items():
        if key in player_data_transposed['Metric'].values:
            player_data_transposed.loc[player_data_transposed['Metric'] == key, 'Value'] = value
        else:
            # Add additional info as a new row if the key is not already in the player's data
            player_data_transposed = pd.concat([player_data_transposed, pd.DataFrame({'Metric': [key], 'Value': [value]})])

    # Set the 'Metric' column as the index so that it doesn't appear as a row in the table
    player_data_transposed.set_index('Metric', inplace=True)

    # Style the DataFrame
    styled_player_data = player_data_transposed.style.set_table_styles(
        [
            {'selector': 'th', 'props': [('background-color', '#ADD8E6'), ('color', 'black'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '5px')]}
        ]
    ).set_properties(**{'background-color': '#F2F2F2', 'color': 'black', 'border-color': 'black', 'text-align': 'center'})

    return styled_player_data


def plot_triple_threat_with_distributions(
    player_name, columns, df, rotate_labels=None, rotation_angles=None,
    min_minutes=200, padding=15, figsize=(12, 8), player_label=None
):
    """
    Creates a radar chart with 1D distributions for a player.

    Parameters:
    - player_name: str, the player's actual name in the dataset.
    - columns: list, metrics to include in the radar chart.
    - df: pandas DataFrame, containing player data.
    - rotate_labels: list, column names to rotate on the radar chart.
    - rotation_angles: list, angles corresponding to the labels to rotate.
    - min_minutes: int, minimum filter for "Minutes played".
    - padding: int, padding for text annotations in the radar chart.
    - figsize: tuple, size of the figure.
    - player_label: str, a custom label for the player's marker in the 1D distributions
      (e.g., "Mystery Player", "Hidden Player", etc.). If None, the actual player name is used.
    """
    # Step 1: Filter players by minimum minutes played
    df_filtered = df[df['Minutes played'] >= min_minutes].copy()

    # Replace infinities with NaN and drop missing data
    df_filtered[columns] = df_filtered[columns].replace([np.inf, -np.inf], np.nan)
    df_filtered.dropna(subset=columns, inplace=True)

    if player_name not in df_filtered['Player'].values:
        print(f"Player {player_name} not found in the filtered DataFrame (minimum {min_minutes} minutes played).")
        return

    # Step 2: Compute percentiles dynamically for the specified columns
    percentiles = {col: df_filtered[col].rank(pct=True) * 100 for col in columns}

    # Step 3: Extract player data and compute percentile scores
    player_data = df_filtered[df_filtered['Player'] == player_name].iloc[0]
    values = [percentiles[col][player_data.name] for col in columns]
    values.append(values[0])  # Close the loop
    categories = columns + [columns[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

    # Step 4: Create figure layout
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.65)  # Adjusted wspace for smaller gap

    # Radar chart on the left
    ax_radar = fig.add_subplot(gs[0], polar=True)
    ax_radar.plot(angles, values, linewidth=2, linestyle='solid', color='blue', label="Percentile")
    ax_radar.fill(angles, values, color='blue', alpha=0.25)

    for i, (label, value) in enumerate(zip(categories[:-1], values[:-1])):
        top_x = 100 - int(value)
        radial_position = 100 + padding
        if rotate_labels and label in rotate_labels:
            rotation_angle = rotation_angles[rotate_labels.index(label)]
            ax_radar.text(
                angles[i], radial_position, f"{label}\n(Top {top_x}%)", fontsize=9,
                ha='center', va='center', rotation=rotation_angle
            )
        else:
            ax_radar.text(
                angles[i], radial_position, f"{label}\n(Top {top_x}%)",
                fontsize=9, ha='center', va='center'
            )

    ax_radar.set_xticks([])
    ax_radar.set_yticks([20, 40, 60, 80, 100])
    ax_radar.set_ylim(0, 100)
    num_players = len(df_filtered)
    ax_radar.set_title(
        f"Triple Threat Radar\nComparison of {num_players} Players (Min {min_minutes} minutes played)",
        fontsize=12, pad=20
    )
    ax_radar.legend(["Percentile"], loc='upper left', bbox_to_anchor=(-0.1, 1.15))

    # Distributions on the right
    gs_right = fig.add_gridspec(len(columns), 1, left=0.55, right=0.95, hspace=0.3)  # Smaller gap between charts
    for i, column in enumerate(columns):
        ax = fig.add_subplot(gs_right[i])
        sns.stripplot(
            x=df_filtered[column], ax=ax, orient="h", color="lightgreen", size=5, alpha=0.7,
            label="_nolegend_"  # Suppress legend for other players
        )

        # Determine label for the player
        custom_label = player_label if player_label else f"{player_name}'s value"

        # Add player's value as a blue dot
        ax.scatter(player_data[column], 0, color='blue', s=100, label=custom_label)
        ax.set_title(column, fontsize=10, pad=10)

        # Remove x-axis label and ticks
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', left=False, labelleft=False)

        # Add a legend only for the player's value (top right of each subplot)
        ax.legend([custom_label], loc='upper right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.show()

def plot_variable_vs_points_with_regression(concatenated_df, x_column, x_label):
    """
    General function to plot any variable against points earned, with a regression line and team names.

    Parameters:
    concatenated_df (DataFrame): The dataframe containing team statistics.
    x_column (str): The column name to plot on the x-axis (e.g., 'PPDA', 'Match Tempo').
    x_label (str): The label for the x-axis (e.g., 'PPDA', 'Match Tempo').
    """

    # Ensure we're only taking unique team stats by removing duplicates
    team_stats = concatenated_df.drop_duplicates().copy()

    # Compute the required metrics: the selected variable (x_column) and final points
    team_metrics = team_stats.groupby('Team').agg({
        x_column: 'mean',  # Average of the selected column
        'Points Earned': 'sum'  # Total points over the season
    }).reset_index()

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Create the scatterplot with a regression line using seaborn
    sns.regplot(x=x_column, y='Points Earned', data=team_metrics,
                scatter_kws={'s': 100}, color='blue', line_kws={'color': 'black'}, ax=ax)

    # Add team names to the plot using an offset in points
    for i, row in team_metrics.iterrows():
        ax.annotate(
            row['Team'],
            xy=(row[x_column], row['Points Earned']),
            xytext=(5, 5),  # Offset in points (x, y)
            textcoords='offset points',
            fontsize=12,
            ha='center'
        )

    # Add labels and title
    ax.set_title(f'{x_label} vs Points Earned - Lithuanian A Lyga 2025', fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Points Earned', fontsize=14)
    ax.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()
