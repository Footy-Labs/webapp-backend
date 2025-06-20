cleaned_wyscout_mapping = {
    'General': [
        'Team', 'Position', 'Age', 'Market value', 'Contract expires',
        'Birth country', 'Passport country',
        'Foot', 'Height', 'Weight', 'On loan'
    ],
    'Performance': [
        'Matches played', 'Minutes played'
    ],
    'Defensive': [
        'Duels per 90', 'Duels won, %',
        'Successful defensive actions per 90', 'Defensive duels per 90',
        'Defensive duels won, %', 'Aerial duels per 90', 'Aerial duels won, %',
        'Sliding tackles per 90', 'PAdj Sliding tackles', 'Shots blocked per 90',
        'Interceptions per 90', 'PAdj Interceptions', 'Fouls per 90',
        'Yellow cards', 'Yellow cards per 90', 'Red cards', 'Red cards per 90'
    ],
    'Attacking': [
        'Successful attacking actions per 90', 'Goals', 'Goals per 90',
        'Non-penalty goals', 'Non-penalty goals per 90', 'xG', 'xG per 90',
        'Head goals', 'Head goals per 90', 'Shots', 'Shots per 90',
        'Shots on target, %', 'Goal conversion, %', 'Assists', 'Assists per 90',
        'xA', 'xA per 90', 'Crosses per 90', 'Accurate crosses, %',
        'Crosses from left flank per 90', 'Accurate crosses from left flank, %',
        'Crosses from right flank per 90', 'Accurate crosses from right flank, %',
        'Crosses to goalie box per 90', 'Dribbles per 90', 'Successful dribbles, %',
        'Offensive duels per 90', 'Offensive duels won, %', 'Touches in box per 90',
        'Progressive runs per 90', 'Received passes per 90', 'Received long passes per 90',
        'Fouls suffered per 90'
    ],
    'Passing': [
        'Passes per 90', 'Accurate passes, %', 'Forward passes per 90',
        'Accurate forward passes, %', 'Back passes per 90', 'Accurate back passes, %',
        'Lateral passes per 90', 'Accurate lateral passes, %',
        'Short / medium passes per 90', 'Accurate short / medium passes, %',
        'Long passes per 90', 'Accurate long passes, %', 'Average pass length, m',
        'Average long pass length, m'
    ],
    'Key Passing': [
        'Shot assists per 90',
        'Second assists per 90', 'Third assists per 90', 'Smart passes per 90',
        'Accurate smart passes, %', 'Key passes per 90',
        'Passes to final third per 90', 'Accurate passes to final third, %',
        'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
        'Through passes per 90', 'Accurate through passes, %',
        'Deep completions per 90', 'Deep completed crosses per 90',
        'Progressive passes per 90', 'Accurate progressive passes, %'
    ],
    'Goalkeeping': [
        'Conceded goals', 'Conceded goals per 90', 'Shots against', 'Shots against per 90',
        'Clean sheets', 'Save rate, %', 'xG against', 'xG against per 90',
        'Prevented goals', 'Prevented goals per 90', 'Back passes received as GK per 90',
        'Exits per 90'
    ],
    'Set Pieces': [
        'Free kicks per 90', 'Direct free kicks per 90', 'Direct free kicks on target, %',
        'Corners per 90', 'Penalties taken', 'Penalty conversion, %'
    ]
}

key_metrics_dict = {
    'Goalkeeper': {
        'Conceded goals per 90': ('Goalkeeping', 'blue'),
        'Clean sheets': ('Goalkeeping', 'blue'),
        'Save rate, %': ('Goalkeeping', 'blue'),
        'Prevented goals per 90': ('Goalkeeping', 'blue'),
        'Exits per 90': ('Goalkeeping', 'blue'),
        'Shots against per 90': ('Goalkeeping', 'blue'),
        'xG against per 90': ('Goalkeeping', 'blue'),
        'Back passes received as GK per 90': ('Passing', 'green'),
        'Passes per 90': ('Passing', 'green'),
        'Accurate passes, %': ('Passing', 'green'),
        'Long passes per 90': ('Passing', 'green'),
        'Accurate long passes, %': ('Passing', 'green')
    },
    'Full Back': {
        'Successful defensive actions per 90': ('Defense', 'blue'),
        'Defensive duels won, %': ('Defense', 'blue'),
        'PAdj Interceptions': ('Defense', 'blue'),
        'PAdj Sliding tackles': ('Defense', 'blue'),
        'xG per 90': ('Attack', 'red'),
        'Goals per 90': ('Attack', 'red'),
        'Assists per 90': ('Attack', 'red'),
        'xA per 90': ('Attack', 'red'),
        'Accurate crosses, %': ('Passing', 'green'),
        'Passes per 90': ('Passing', 'green'),
        'Accurate passes, %': ('Passing', 'green'),
        'Accurate forward passes, %': ('Passing', 'green'),
        'Key passes per 90': ('Passing', 'green')
    },
    'Centre Back': {
        'Interceptions per 90': ('Defense', 'blue'),
        'Duels won, %': ('Defense', 'blue'),
        'Aerial duels won, %': ('Defense', 'blue'),
        'Shots blocked per 90': ('Defense', 'blue'),
        'Fouls per 90': ('Defense', 'blue'),
        'Accurate passes, %': ('Passing', 'green'),
        'Accurate passes to final third per 90': ('Passing', 'green'),
        'Through passes per 90': ('Passing', 'green'),
        'Goals per 90': ('Attack', 'red'),
        'Assists per 90': ('Attack', 'red'),
    },
    'Defensive Midfielder': {
        'Interceptions per 90': ('Defense', 'blue'),
        'Sliding tackles per 90': ('Defense', 'blue'),
        'Aerial duels per 90': ('Defense', 'blue'),
        'Fouls suffered per 90': ('Defense', 'blue'),
        'Progressive passes per 90': ('Passing', 'green'),
        'Key passes per 90': ('Passing', 'green'),
        'Passes to final third per 90': ('Passing', 'green'),
        'Through passes per 90': ('Passing', 'green'),
        'Progressive runs per 90': ('Attack', 'red'),
        'Received passes per 90': ('Attack', 'red'),
        'Touches in box per 90': ('Attack', 'red')
    },
    'Central Midfielder': {
        'Successful defensive actions per 90': ('Defense', 'blue'),
        'Defensive duels won, %': ('Defense', 'blue'),
        'PAdj Interceptions': ('Defense', 'blue'),
        'Fouls per 90': ('Defense', 'blue'),
        'Accurate passes, %': ('Passing', 'green'),
        'Accurate passes to penalty area per 90': ('Passing', 'green'),
        'Forward passes per 90': ('Passing', 'green'),
        'Accurate forward passes, %': ('Passing', 'green'),
        'Assists per 90': ('Passing', 'green'),
        'xA per 90': ('Passing', 'green'),
        'xG per 90': ('Attack', 'red'),
        'Successful dribbles per 90': ('Attack', 'red')
    },
    'Attacking Midfielder': {
        'Successful defensive actions per 90': ('Defense', 'blue'),
        'Defensive duels won, %': ('Defense', 'blue'),
        'PAdj Interceptions': ('Defense', 'blue'),
        'Fouls per 90': ('Defense', 'blue'),
        'Accurate passes, %': ('Passing', 'green'),
        'Accurate passes to penalty area per 90': ('Passing', 'green'),
        'Accurate smart passes per 90': ('Passing', 'green'),
        'Assist Overperformance': ('Passing', 'green'),
        'Goals per 90': ('Attack', 'red'),
        'Successful dribbles per 90': ('Attack', 'red')
    },
    'Winger': {
        'Goals per 90': ('Attack', 'red'),
        'Non-penalty goals per 90': ('Attack', 'red'),
        'xG per 90': ('Attack', 'red'),
        'Shots per 90': ('Attack', 'red'),
        'Shots on target per 90': ('Attack', 'red'),
        'Successful dribbles per 90': ('Attack', 'red'),
        'Assists per 90': ('Passing', 'green'),
        'xA per 90': ('Passing', 'green'),
        'Passes per 90': ('Passing', 'green'),
        'Forward passes per 90': ('Passing', 'green'),
        'Accurate passes, %': ('Passing', 'green')
    },
    'Centre Forward': {
        'Non-penalty goals per 90': ('Attack', 'red'),
        'xG per 90': ('Attack', 'red'),
        'Shots on target per 90': ('Attack', 'red'),
        'Touches in box per 90': ('Attack', 'red'),
        'Successful dribbles per 90': ('Attack', 'red'),
        'Assists per 90': ('Passing', 'green'),
        'xA per 90': ('Passing', 'green'),
        'Passes per 90': ('Passing', 'green'),
        'Accurate passes, %': ('Passing', 'green'),
        'Key passes per 90': ('Passing', 'green'),
        'Interceptions per 90': ('Defense', 'blue'),
        'Duels won, %': ('Defense', 'blue'),
    }
    # Add key metrics for other positions as needed
}

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



