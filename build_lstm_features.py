import re
import numpy as np
import pandas as pd

import re
import numpy as np
import pandas as pd
 
POLLS_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/raw_polls.csv"
ELECTIONS_CSV = "elections_clean_with_polls.csv"
OUT_NPZ = "lstm_data.npz"
TEST_CANDIDATES_CSV = "test_candidates.csv"
TEST_YEAR = 2022
 
SEQUENCE_WEEKS = 12
MIN_POLLS_PER_CANDIDATE = 1
MAX_DAYS_BEFORE = SEQUENCE_WEEKS * 7
 
 
def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).upper().strip()
    if "," in name:
        parts = name.split(",", 1)
        name = parts[1].strip() + " " + parts[0].strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name
 
 
def parse_race(race_str):
    """Parse FiveThirtyEight race strings like '2022_Sen-G_PA' or '2022_H-G_PA-01'"""
    parts = str(race_str).split("_")
    year = int(parts[0])
    office = "S" if "Sen" in parts[1] else "H"
    if office == "H":
        sd = parts[2]
        if "-" in sd:
            state, district = sd.split("-")
        else:
            state, district = sd, "00"
        district = district.zfill(2)
    else:
        state = parts[2]
        district = "00"
    return year, state, office, district
 
 
def load_polls():
    """Load 538 raw_polls.csv and expand into one row per (poll, candidate).
    Each original poll has cand1 and cand2, so this becomes two rows per poll.
    """
    print(f"Loading polls from {POLLS_URL}...")
    raw = pd.read_csv(POLLS_URL, encoding="utf-8", low_memory=False)
    print(f"{len(raw)} poll rows loaded")
 
    cand1 = raw[["race", "polldate", "electiondate",
                 "cand1_name", "cand1_party", "cand1_pct"]].rename(columns={
        "cand1_name": "candidate_name",
        "cand1_party": "party",
        "cand1_pct": "poll_pct",
    })
    cand2 = raw[["race", "polldate", "electiondate",
                 "cand2_name", "cand2_party", "cand2_pct"]].rename(columns={
        "cand2_name": "candidate_name",
        "cand2_party": "party",
        "cand2_pct": "poll_pct",
    })
    polls = pd.concat([cand1, cand2], ignore_index=True)
    print(f"{len(polls)} rows after expanding cand1 + cand2")
 
    polls = polls[polls["party"].isin(["DEM", "REP"])].copy()
    polls = polls.dropna(subset=["poll_pct", "candidate_name"])
    print(f"{len(polls)} rows after DEM/REP + non-null filter")
 
    polls["candidate_normalized"] = polls["candidate_name"].apply(normalize_name)
    polls["poll_date"] = pd.to_datetime(polls["polldate"], errors="coerce")
    polls["election_date"] = pd.to_datetime(polls["electiondate"], errors="coerce")
    polls = polls.dropna(subset=["poll_date", "election_date"])
 
    polls["days_before_election"] = (
        polls["election_date"] - polls["poll_date"]
    ).dt.days
 
    polls = polls[polls["days_before_election"] >= 0]
    polls = polls[polls["days_before_election"] <= MAX_DAYS_BEFORE]
 
    parsed = polls["race"].apply(parse_race)
    polls["year"] = [p[0] for p in parsed]
    polls["state"] = [p[1] for p in parsed]
    polls["office"] = [p[2] for p in parsed]
    polls["district"] = [p[3] for p in parsed]
 
    print(f"{len(polls)} final polls in the {MAX_DAYS_BEFORE}-day window")
    print(f"Party balance: {polls['party'].value_counts().to_dict()}")
 
    return polls
 
 
def build_sequence(candidate_polls, sequence_weeks=SEQUENCE_WEEKS):
    """Build a weekly polling sequence for one candidate.
 
    Returns (values, mask) arrays of length sequence_weeks where index 0 is the
    oldest week and index -1 is the election week. Values are forward filled.
    """
    values = np.zeros(sequence_weeks, dtype=np.float32)
    mask = np.zeros(sequence_weeks, dtype=np.float32)
 
    cp = candidate_polls.copy()
    cp["week"] = cp["days_before_election"] // 7
    cp = cp[cp["week"] < sequence_weeks]
 
    weekly = cp.groupby("week")["poll_pct"].mean()
    for w, v in weekly.items():
        values[int(w)] = float(v)
        mask[int(w)] = 1.0
 
    last_known = None
    for w in range(sequence_weeks - 1, -1, -1):
        if mask[w] == 1.0:
            last_known = values[w]
        elif last_known is not None:
            values[w] = last_known
 
    values = values[::-1].copy()
    mask = mask[::-1].copy()
    return values, mask, int(weekly.shape[0])
 
 
def main():
    polls = load_polls()
 
    print(f"\nLoading {ELECTIONS_CSV}...")
    elections = pd.read_csv(ELECTIONS_CSV)
 
    # Force column types to match the poll side of the join to fix a no match issue due to CSV reading being different than what was written
    elections["year"] = elections["year"].astype(int)
    elections["district"] = elections["district"].astype(str).str.zfill(2)
    elections["state"] = elections["state"].astype(str)
    elections["office"] = elections["office"].astype(str)
    elections["party"] = elections["party"].astype(str)
 
    elections["name_normalized"] = elections["candidate_mit"].apply(normalize_name)
 
    # Drop duplicate candidate rows that I noticed in the CSV after merging
    before = len(elections)
    elections = elections.drop_duplicates(
        subset=["year", "state", "office", "district", "party", "name_normalized"],
        keep="first",
    )
    print(f"Dropped {before - len(elections)} duplicate candidate rows")
 
    before = len(elections)
    elections = elections[elections["incumbency"] != "unknown"].copy()
    print(f"{len(elections)}/{before} candidates survive FEC match filter")
 
    poll_keys = ["year", "state", "office", "district", "party", "candidate_normalized"]
    polls_by_candidate = polls.groupby(poll_keys)
 
    sequences, masks, static_rows, targets, meta = [], [], [], [], []
    n_dropped_nopolls = 0
    n_dropped_fewpolls = 0
 
    # Had Claude's help in implementing this loop to build the sequences and handle the various edge cases, 
    # especially in terms of the filtering and the forward filling of the sequences.
    for _, row in elections.iterrows():
        key = (row["year"], row["state"], row["office"], row["district"],
               row["party"], row["name_normalized"])
        try:
            cand_polls = polls_by_candidate.get_group(key)
        except KeyError:
            n_dropped_nopolls += 1
            continue
 
        seq, mask, n_obs_weeks = build_sequence(cand_polls)
        if n_obs_weeks < MIN_POLLS_PER_CANDIDATE:
            n_dropped_fewpolls += 1
            continue
 
        sequences.append(seq)
        masks.append(mask)
        static_rows.append(row)
        targets.append(row["won"])
        meta.append((row["year"], row["state"], row["office"],
                     row["district"], row["party"], row["candidate_mit"]))
 
    print(f"\ndropped {n_dropped_nopolls} candidates with no polls in window")
    print(f"dropped {n_dropped_fewpolls} candidates with < {MIN_POLLS_PER_CANDIDATE} observed weeks")
    print(f"kept {len(sequences)} candidates")
 
    if not sequences:
        raise RuntimeError("No candidates survived filtering. BIG ERROR SHOULDN'T HAPPEN.")
 
    sequences = np.stack(sequences)
    masks = np.stack(masks)
    static_df = pd.DataFrame(static_rows).reset_index(drop=True)
    targets = np.asarray(targets, dtype=np.float32)
    years = static_df["year"].to_numpy()
 
    static_df["incumbent_flag"] = (static_df["incumbency"] == "incumbent").astype(int)
    static_df["open_flag"] = (static_df["incumbency"] == "open").astype(int)
 
    static_features = [
        "TTL_RECEIPTS",
        "TTL_INDIV_CONTRIB",
        "OTHER_POL_CMTE_CONTRIB",
        "POL_PTY_CONTRIB",
        "is_midterm",
        "incumbent_flag",
        "open_flag",
    ]
    if "partisan" in static_df.columns:
        static_features.append("partisan")
 
    X_static = static_df[static_features].fillna(0).to_numpy(dtype=np.float32)
 
    meta_arr = np.array(
        meta,
        dtype=[("year", "i4"), ("state", "U4"), ("office", "U1"),
               ("district", "U2"), ("party", "U3"), ("candidate", "U128")]
    )
 
    # I love printing stats!  I love printing stats!  Thanks Claude for making all of the prints look nice so I don't have to spend hours formatting them myself!
    party_counts = pd.Series(meta_arr["party"]).value_counts().to_dict()
    print(f"\nParty balance in final dataset: {party_counts}")
 
    race_counts = pd.DataFrame({
        "year": meta_arr["year"], "state": meta_arr["state"],
        "office": meta_arr["office"], "district": meta_arr["district"],
    }).value_counts()
    paired = (race_counts == 2).sum()
    single = (race_counts == 1).sum()
    print(f"Races with both candidates: {paired}")
    print(f"Races with only one candidate: {single}")
 
    print(f"\nClass balance: win rate = {targets.mean():.3f}")
    print(f"Year distribution:")
    for y, c in sorted(pd.Series(years).value_counts().to_dict().items()):
        print(f"  {y}: {c}")
 
    np.savez(
        OUT_NPZ,
        sequences=sequences,
        masks=masks,
        X_static=X_static,
        y=targets,
        years=years,
        static_feature_names=np.array(static_features),
        meta=meta_arr,
    )
    print(f"\nWrote {OUT_NPZ}")
    print(f"  sequences: {sequences.shape}")
    print(f"  masks:     {masks.shape}")
    print(f"  X_static:  {X_static.shape} features={static_features}")
    print(f"  y:         {targets.shape}")
 
    # Save test candidates for evaluate.py
    test_mask_npz = years == TEST_YEAR
    test_candidates = pd.DataFrame({
        "year": meta_arr["year"][test_mask_npz],
        "state": meta_arr["state"][test_mask_npz],
        "office": meta_arr["office"][test_mask_npz],
        "district": meta_arr["district"][test_mask_npz],
        "party": meta_arr["party"][test_mask_npz],
        "candidate": meta_arr["candidate"][test_mask_npz],
    })
    test_candidates.to_csv(TEST_CANDIDATES_CSV, index=False)
    print(f"Wrote {TEST_CANDIDATES_CSV} with {len(test_candidates)} rows")
 
 
if __name__ == "__main__":
    main()




"""
FILE DESCRIPTION
Build per-candidate polling sequences for the LSTM.

Reads FiveThirtyEight's raw_polls.csv and joins against elections_clean.csv
(the output of process_data.py). For each candidate in each race, produces a
weekly bucketed polling sequence covering the last SEQUENCE_WEEKS weeks before
election day, forward filled through gaps with a mask channel indicating which
weeks were actually observed versus carried forward.

Candidates with less than than MIN_POLLS_PER_CANDIDATE weekly polls are dropped
Candidates with no FEC match are dropped

Output - lstm_data.npz with:
  sequences  (N, SEQUENCE_WEEKS)   float weekly polling averages
  masks      (N, SEQUENCE_WEEKS)   float 1.0 = observed, 0.0 = forward-filled
  X_static   (N, D)                float static features
  y          (N,)                  float win=1 / loss=0
  years      (N,)                  int   election year
  static_feature_names             array of feature name strings
  meta                             structured array with race identifiers
"""