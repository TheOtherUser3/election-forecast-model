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

# Partisanship
pres = pd.read_csv("/content/1976-2020-president.csv")
pres = pres[pres["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])]
pres["vote_share"] = pres["candidatevotes"] / pres["totalvotes"]

# State-level results
state_results = pres.pivot_table(
    index=["year", "state_po"],
    columns="party_simplified",
    values="vote_share"
).reset_index()
state_results["dem_margin"] = (
    state_results["DEMOCRAT"].fillna(0) - state_results["REPUBLICAN"].fillna(0)
)
# National popular vote
national = pres.groupby(["year", "party_simplified"])["candidatevotes"].sum().unstack()

national["dem_share"] = national["DEMOCRAT"] / (
    national["DEMOCRAT"] + national["REPUBLICAN"]
)

national["rep_share"] = 1 - national["dem_share"]

national["national_margin"] = national["dem_share"] - national["rep_share"]

national = national.reset_index()[["year", "national_margin"]]

# combine
pvi = pd.merge(state_results, national, on="year")

pvi["partisan"] = pvi["dem_margin"] - pvi["national_margin"]

pvi_midterm = pvi.copy()
pvi_midterm["year"] = pvi_midterm["year"] + 2

pvi_full = pd.concat([pvi, pvi_midterm], ignore_index=True)

pvi_full = pvi_full[["year", "state_po", "partisan"]]

# Merge into final dataframe
final_df = final_df.merge(
    pvi_full,
    left_on=["year", "state"],
    right_on=["year", "state_po"],
    how="left"
)

# Clean up column name
final_df = final_df.drop(columns=["state_po"])

print(f"Polls matched: {final_df['poll_available'].sum()}/{len(final_df)}")
print(final_df[['year', 'state', 'office', 'district', 'party', 'candidate_mit', 'poll_avg', 'poll_available']].head())

final_df.to_csv("elections_clean_with_polls.csv", index=False)
print(f"Saved final dataset with polls: {len(final_df)} rows")

final_df
