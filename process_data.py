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
            "candidate_fec": None, "vote_pct_fec": None, "CAND_ID": None
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

print("\nWin rate by party:")
print(merged.groupby("party")["won"].mean())

print("\nIncumbency distribution:")
print(merged["incumbency"].value_counts())

print("\nMidterm vs presidential year split:")
print(merged["is_midterm"].value_counts())

print("\nSample rows:")
print(merged[["year", "state", "office", "district", "party",
              "incumbency", "TTL_RECEIPTS", "vote_share", "won", "is_midterm"]].head(10))


# CURRENT FINAL DATASET AFTER PROCESSING: merged (pandas DataFrame variable type for easy joins)
# Each row is one candidate in one race. Only Democrat and Republican general
# election candidates are included. Both midterm and presidential year elections
# are included from 1976-2024 (house) and 1976-2022 (senate).
#
# Columns:
#   year            - election year (int)
#   state           - two-letter state abbreviation "AL" "NY"
#   office          - "H" for House, "S" for Senate
#   district        - zero-padded district number string. "11" for District 11
#   party           - "DEM" or "REP"
#   candidate_mit   - candidate name as it appears in MIT election results
#   vote_share      - fraction of total votes received (float, 0-1). None for 2022 senate rows since we only need to know win vs loss for test set
#   totalvotes      - total votes cast in the race. None for 2022 senate rows.
#   won             - 1 if this candidate won the general election, 0 if they lost
#   is_midterm      - 1 if midterm year, 0 if presidential year
#   name_normalized - cleaned/normalized version of candidate_mit used for FEC joining
#   incumbency      - "incumbent", "challenger", "open", or "unknown" if FEC had no match
#   TTL_RECEIPTS    - total money raised by candidate (float). 0 if FEC had no match.
#   TTL_INDIV_CONTRIB     - total individual contributions (float)
#   OTHER_POL_CMTE_CONTRIB - contributions from other political committees (float)
#   POL_PTY_CONTRIB - contributions from party committees (float) (probably will just want to sum all of these up at end?)
#   candidate_fec   - candidate name as it appears in FEC data. None if no FEC match.
#   vote_pct_fec    - general election vote percentage from FEC (only available pre-2012)  Completely unecessary, just an artifict we can remove at the end


# SAVE 

merged.to_csv("elections_clean.csv", index=False)
print(f"\nSaved to elections_clean.csv ({len(merged)} rows)")

unmatched = merged[merged["incumbency"] == "unknown"]
print(unmatched[unmatched["year"] == 2022][["year", "state", "office", "district", "party", "candidate_mit"]].head(20))
