TEAM_NAME_MAP = {
    "Arsenal": "Arsenal FC",
    "Arsenal FC": "Arsenal FC",

    "Aston Villa": "Aston Villa FC",
    "Aston Villa FC": "Aston Villa FC",

    "Bournemouth": "AFC Bournemouth",
    "AFC Bournemouth": "AFC Bournemouth",

    "Brentford": "Brentford FC",
    "Brentford FC": "Brentford FC",

    "Brighton": "Brighton & Hove Albion FC",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion FC",

    "Chelsea": "Chelsea FC",
    "Chelsea FC": "Chelsea FC",

    "Crystal Palace": "Crystal Palace FC",
    "Crystal Palace FC": "Crystal Palace FC",

    "Everton": "Everton FC",
    "Everton FC": "Everton FC",

    "Fulham": "Fulham FC",
    "Fulham FC": "Fulham FC",

    "Liverpool": "Liverpool FC",
    "Liverpool FC": "Liverpool FC",

    "Man City": "Manchester City FC",
    "Manchester City": "Manchester City FC",
    "Manchester City FC": "Manchester City FC",

    "Man Utd": "Manchester United FC",
    "Manchester United": "Manchester United FC",
    "Manchester United FC": "Manchester United FC",

    "Newcastle": "Newcastle United FC",
    "Newcastle United": "Newcastle United FC",
    "Newcastle United FC": "Newcastle United FC",

    "Nottingham Forest": "Nottingham Forest FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Nottingham Forest FC": "Nottingham Forest FC",

    "Leeds": "Leeds United FC",
    "Leeds United": "Leeds United FC",
    "Leeds United FC": "Leeds United FC",

    "Southampton": "Southampton FC",
    "Southampton FC": "Southampton FC",

    "Spurs": "Tottenham Hotspur FC",
    "Tottenham": "Tottenham Hotspur FC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Tottenham Hotspur FC": "Tottenham Hotspur FC",

    "West Ham": "West Ham United FC",
    "West Ham United": "West Ham United FC",
    "West Ham United FC": "West Ham United FC",

    "Wolves": "Wolverhampton Wanderers FC",
    "Wolverhampton": "Wolverhampton Wanderers FC",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers FC",

    "Burnley": "Burnley FC",
    "Burnley FC": "Burnley FC",

    "Sunderland": "Sunderland AFC",
    "Sunderland AFC": "Sunderland AFC",
}


def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)
