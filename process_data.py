import pandas as pd
import os
import glob
import re
from rapidfuzz import process, fuzz


# COLUMN NAMES FOR WEBALL FILES
weball_cols = [
    "CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION",
    "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
    "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS",
    "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB",
    "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION",
    "RUN_ELECTION", "GEN_ELECTION", "GEN_ELECTION_PRECENT", "OTHER_POL_CMTE_CONTRIB",
    "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS"
]

# LOAD ALL WEBALL FILES 
# Folder names: weball00 = 2000, weball98 = 1998, weball06 = 2006 etc.

weball_frames = []

for folder in glob.glob("Data/weball*"):
    # Extract year from folder name
    suffix = re.search(r'weball(\d{2})', folder)
    if not suffix:
        continue
    yr = int(suffix.group(1))
    # 00-26 = 2000-2026, 76-99 = 1976-1999
    year = 2000 + yr if yr <= 26 else 1900 + yr

    txt_path = os.path.join(folder, f"weball{suffix.group(1)}.txt")
    if not os.path.exists(txt_path):
        continue

    df = pd.read_csv(
        txt_path, sep="|", names=weball_cols,
        encoding="latin-1", on_bad_lines="skip"
    )
    df["year"] = year
    weball_frames.append(df)

weball = pd.concat(weball_frames, ignore_index=True)
print(f"Raw weball records: {len(weball)}")

# FILTER TO WHAT WE NEED

# House and Senate only (CAND_ID starts with H or S)
weball = weball[weball["CAND_ID"].str[0].isin(["H", "S"])]

# Democrat and Republican only
weball = weball[weball["CAND_PTY_AFFILIATION"].isin(["DEM", "REP"])]

# Only candidates who actually received money since those who didn't are unlikely to have been major candidates
weball = weball[weball["TTL_RECEIPTS"] > 0]

# Get office from CAND_ID first character
weball["office"] = weball["CAND_ID"].str[0]  # H or S

# Clean district
weball["CAND_OFFICE_DISTRICT"] = (
    pd.to_numeric(weball["CAND_OFFICE_DISTRICT"], errors="coerce")
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(2)
)
# Incumbency: I=incumbent, C=challenger, O=open seat
weball["incumbency"] = weball["CAND_ICI"].map({
    "I": "incumbent",
    "C": "challenger",
    "O": "open"
})

# Did they win?
weball["won"] = (weball["GEN_ELECTION"] == "W").astype(int)

# Midterm flag: presidential years are divisible by 4
weball["is_midterm"] = (weball["year"] % 4 != 0).astype(int)

# Clean fundraising data: convert to numbers, fill missing with 0
money_cols = ["TTL_RECEIPTS", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB"]
for col in money_cols:
    weball[col] = pd.to_numeric(weball[col], errors="coerce").fillna(0)

# Get rid of unecessary columns
fec_clean = weball[[
    "year", "CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION",
    "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "office",
    "incumbency", "TTL_RECEIPTS", "TTL_INDIV_CONTRIB",
    "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB",
    "GEN_ELECTION_PRECENT", "won", "is_midterm"
]].rename(columns={
    "CAND_PTY_AFFILIATION": "party",
    "CAND_OFFICE_ST": "state",
    "CAND_OFFICE_DISTRICT": "district",
    "CAND_NAME": "candidate_fec",
    "GEN_ELECTION_PRECENT": "vote_pct_fec"
})

print(f"Filtered FEC records: {len(fec_clean)}")
print(fec_clean.head())


# LOAD HOUSE RESULTS FROM MIT

house = pd.read_csv("Data/house_results/1976-2024-house.tab", sep=",", encoding="utf-8", low_memory=False)
house = house[house["stage"] == "GEN"]
house = house[house["writein"] == False]
house = house[house["party"].isin(["DEMOCRAT", "REPUBLICAN"])] 
house["party"] = house["party"].map({"DEMOCRAT": "DEM", "REPUBLICAN": "REP"})
house["district"] = house["district"].astype(str).str.zfill(2)
house["vote_share"] = house["candidatevotes"] / house["totalvotes"]
house["is_midterm"] = (house["year"] % 4 != 0).astype(int)

# THANK YOU CLAUDE AI FOR THIS LINE
house["won"] = house.groupby(["year", "state_po", "district"])["candidatevotes"].transform(
    lambda x: (x == x.max())
).astype(int)

house_clean = house[[
    "year", "state_po", "district", "candidate", "party",
    "vote_share", "totalvotes", "won", "is_midterm"
]].rename(columns={"state_po": "state", "candidate": "candidate_mit"})
house_clean["office"] = "H"


# LOAD SENATE RESULTS FROM MIT

senate = pd.read_csv("Data/senate_results/1976-2020-senate.csv", encoding="utf-8")
senate = senate[senate["stage"] == "gen"]
senate = senate[senate["writein"] == False]
senate = senate[senate["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])]

senate["party"] = senate["party_simplified"].map({"DEMOCRAT": "DEM", "REPUBLICAN": "REP"})
senate["district"] = "00"
senate["vote_share"] = senate["candidatevotes"] / senate["totalvotes"]
senate["is_midterm"] = (senate["year"] % 4 != 0).astype(int)

senate["won"] = senate.groupby(["year", "state_po"])["candidatevotes"].transform(
    lambda x: (x == x.max())
).astype(int)

senate_clean = senate[[
    "year", "state_po", "district", "candidate", "party",
    "vote_share", "totalvotes", "won", "is_midterm"
]].rename(columns={"state_po": "state", "candidate": "candidate_mit"})
senate_clean["office"] = "S"

def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    # Flip LAST, FIRST to FIRST LAST 
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    # remove punctuation and extra spaces
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

# LOAD 2022 SENATE RESULTS (MIT doesn't have these so we made our own)
senate_2022 = pd.read_csv("Data/senate_results/senate_2022.csv", skipinitialspace=True).rename(columns={"name": "candidate_mit"})
senate_2022["district"] = senate_2022["district"].astype(str).str.zfill(2)
senate_2022["is_midterm"] = 1
senate_2022["vote_share"] = None
senate_2022["totalvotes"] = None
senate_2022["name_normalized"] = senate_2022["candidate_mit"].apply(normalize_name)
# COMBINE HOUSE AND SENATE 

mit_results = pd.concat([house_clean, senate_clean, senate_2022], ignore_index=True)
print(f"\nMIT results records: {len(mit_results)}")


# JOIN FEC ONTO MIT RESULTS THANK YOU CS460
# Join key: year + state + office + district + party
# Usually exactly one Dem and one Rep per race, so works well enough



fec_clean["name_normalized"] = fec_clean["candidate_fec"].apply(normalize_name)
mit_results["name_normalized"] = mit_results["candidate_mit"].apply(normalize_name)


# THANKS TO CLAUDE AI FOR TEACHING ME FUZZY MATCHING
fec_lookup = fec_clean.groupby(["year", "state", "office", "district", "party"])

def get_best_fec_match(row):
    key = (row["year"], row["state"], row["office"], row["district"], row["party"])
    try:
        candidates = fec_lookup.get_group(key)
    except KeyError:
        return pd.Series({
            "incumbency": None, "TTL_RECEIPTS": 0, "TTL_INDIV_CONTRIB": 0,
            "OTHER_POL_CMTE_CONTRIB": 0, "POL_PTY_CONTRIB": 0,
            "candidate_fec": None, "vote_pct_fec": None
        })
    result = process.extractOne(
        row["name_normalized"],
        candidates["name_normalized"].tolist(),
        scorer=fuzz.token_sort_ratio,
        score_cutoff=80
    )
    if result:
        match = candidates[candidates["name_normalized"] == result[0]].iloc[0]
        return match[["incumbency", "TTL_RECEIPTS", "TTL_INDIV_CONTRIB",
                       "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB",
                       "candidate_fec", "vote_pct_fec"]]
    return pd.Series({
        "incumbency": None, "TTL_RECEIPTS": 0, "TTL_INDIV_CONTRIB": 0,
        "OTHER_POL_CMTE_CONTRIB": 0, "POL_PTY_CONTRIB": 0,
        "candidate_fec": None, "vote_pct_fec": None
    })

print("Running fuzzy match, this will take a minute...")
fec_matched = mit_results.apply(get_best_fec_match, axis=1)
merged = pd.concat([mit_results, fec_matched], axis=1)

# Check match rate
matched = (merged["incumbency"].notna()).sum()
total = len(merged)
print(f"FEC match rate: {matched}/{total} ({100*matched/total:.2f}%)")

# Fill missing incumbency as unknown
merged["incumbency"] = merged["incumbency"].fillna("unknown")

# Fill missing fundraising with 0 (good enough)
for col in ["TTL_RECEIPTS", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB"]:
    merged[col] = merged[col].fillna(0)


# MAKE SURE NOTHING LOOKS BROKEN

print("\nYear distribution:")
print(merged["year"].value_counts().sort_index())

print("\nWin rate by party (should be close to 50/50):")
print(merged.groupby("party")["won"].mean())

print("\nIncumbency distribution:")
print(merged["incumbency"].value_counts())

print("\nMidterm vs presidential year split:")
print(merged["is_midterm"].value_counts())

print("\nSample rows:")
print(merged[["year", "state", "office", "district", "party",
              "incumbency", "TTL_RECEIPTS", "vote_share", "won", "is_midterm"]].head(10))


# SAVE 

merged.to_csv("elections_clean.csv", index=False)
print(f"\nSaved to elections_clean.csv ({len(merged)} rows)")

unmatched = merged[merged["incumbency"] == "unknown"]
print(unmatched[unmatched["year"] == 2022][["year", "state", "office", "district", "party", "candidate_mit"]].head(20))


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
pres = pd.read_csv("Data/1976-2020-president.csv")
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
