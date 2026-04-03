import pandas as pd

polls_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv"
polls = pd.read_csv(polls_url, encoding="utf-8", low_memory=False)

print("Columns in polls CSV:")
print(polls.columns.tolist())

import pandas as pd
import re

polls_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv"
polls = pd.read_csv(polls_url, encoding="utf-8", low_memory=False)

# Keep only DEM and REP candidates
polls = polls[polls['cand1_party'].isin(['DEM', 'REP'])]

# Normalize candidate names
def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    # Flip LAST, FIRST to FIRST LAST
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    # Remove punctuation and extra spaces
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

polls['candidate_normalized'] = polls['cand1_name'].apply(normalize_name)

# Convert polldate/electiondate to datetime
polls['start_date'] = pd.to_datetime(polls['polldate'], errors='coerce')
polls['end_date'] = pd.to_datetime(polls['electiondate'], errors='coerce')

polls = polls[polls['cand1_party'].isin(['DEM','REP'])]

# Get latest poll per race + candidate
latest_poll = (
    polls.sort_values(['race', 'end_date'])
    .groupby(['race', 'candidate_normalized'])
    .tail(1)
    .rename(columns={'cand1_pct': 'poll_avg', 'cand1_party':'party'})  # <-- fix here
)

# Parse race string into year, state, office, district
def parse_race(race_str):
    parts = race_str.split("_")
    year = int(parts[0])
    office = 'S' if 'Sen' in parts[1] else 'H'
    if office == 'H':
        state_district = parts[2]
        if "-" in state_district:
            state, district = state_district.split("-")
        else:
            state = state_district
            district = "00"
        district = district.zfill(2)
    else:
        state = parts[2]
        district = "00"
    return pd.Series([year, state, office, district])

latest_poll[['year', 'state', 'office', 'district']] = latest_poll['race'].apply(parse_race)
latest_poll["name_normalized"] = latest_poll["candidate_normalized"]

# Normalize partner dataset candidate names
merged["name_normalized"] = merged["candidate_mit"].apply(normalize_name)

# Merge polls
final_df = pd.merge(
    merged,
    latest_poll[['year', 'state', 'office', 'district', 'party', 'name_normalized', 'poll_avg']],
    how='left',  # Keep all rows from merged
    on=['year', 'state', 'office', 'district', 'party', 'name_normalized']
)

# Fill missing poll averages with 0
final_df['poll_avg'] = final_df['poll_avg'].fillna(0)
# Create a Boolean for poll availability
final_df['poll_available'] = final_df['poll_avg'] > 0

# Partisanship -- I think this works?
# Compute partisanship for House
# Margin = DEM - REP per year/state/district
house_margin = mit_results[mit_results['office'] == 'H'].pivot_table(
    index=['year', 'state', 'district'],
    columns='party',
    values='vote_share'
).reset_index()

house_margin['partisan'] = (house_margin['DEM'].fillna(0) - house_margin['REP'].fillna(0))

# Compute partisanship for Senate
# Margin = DEM - REP per year/state
senate_margin = mit_results[mit_results['office'] == 'S'].pivot_table(
    index=['year', 'state'],
    columns='party',
    values='vote_share'
).reset_index()

senate_margin['district'] = '00'
senate_margin['partisan'] = (senate_margin['DEM'].fillna(0) - senate_margin['REP'].fillna(0))

# Combine House and Senate margins
partisan_margin = pd.concat([house_margin[['year', 'state', 'district', 'partisan']],
                             senate_margin[['year', 'state', 'district', 'partisan']]], 
                            ignore_index=True)

# Merge into final dataframe
final_df = final_df.merge(
    partisan_margin,
    on=['year', 'state', 'district'],
    how='left'
)

print(f"Polls matched: {final_df['poll_available'].sum()}/{len(final_df)}")
print(final_df[['year', 'state', 'office', 'district', 'party', 'candidate_mit', 'poll_avg', 'poll_available']].head())

final_df.to_csv("elections_clean_with_polls.csv", index=False)
print(f"Saved final dataset with polls: {len(final_df)} rows")

final_df.head()
