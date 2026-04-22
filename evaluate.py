import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
import lstm_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

DATA_FILE = "lstm_data.npz"
TEST_CANDIDATES_FILE = "test_candidates.csv"
TEST_YEAR = 2022


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def load_test_candidates():
    """Load the list of 2022 test candidates."""
    return pd.read_csv(TEST_CANDIDATES_FILE)


def filter_to_test_set(df, key_cols=("year", "state", "office", "district", "party", "candidate")):
    """ Given a dataframe with prediction + identity columns, keep only rows
    whose identity matches the test candidate list.

    Normalizes types and zero pads district to avoid CSV issues that plagued the poll sequence building in previous builds.
    """
    test_cands = load_test_candidates()
    df = df.copy()
    test_cands = test_cands.copy()

    for col in key_cols:
        if col == "district":
            # Zero pad to width 2 on both sides so "0" matches "00"
            df[col] = df[col].astype(str).str.zfill(2)
            test_cands[col] = test_cands[col].astype(str).str.zfill(2)
        elif col == "year":
            df[col] = df[col].astype(int)
            test_cands[col] = test_cands[col].astype(int)
        else:
            df[col] = df[col].astype(str).str.strip()
            test_cands[col] = test_cands[col].astype(str).str.strip()

    return df.merge(test_cands, on=list(key_cols), how="inner")


def naive_poll_baseline():
    """For each candidate in the LSTM test set, predict win probability from
    final week poll.  final_poll / (final_poll + opponent_final_poll) confidence.
    """
    # allow_pickle=True because the meta field is an array of dicts, which npz doesn't support (Thanks for fixing this Claude)
    data = np.load(DATA_FILE, allow_pickle=True)
    sequences = data["sequences"]
    years = data["years"]
    y = data["y"].astype(int)
    meta = data["meta"]

    final_polls = sequences[:, -1]
    race_keys = np.array([
        f"{m['year']}_{m['state']}_{m['office']}_{m['district']}"
        for m in meta
    ])

    probs = np.zeros(len(final_polls), dtype=np.float64)
    for i in range(len(final_polls)):
        same_race = (race_keys == race_keys[i])
        same_race[i] = False
        if same_race.any():
            opp_poll = final_polls[same_race].max()
            total = final_polls[i] + opp_poll
            probs[i] = final_polls[i] / total if total > 0 else 0.5
        else:
            probs[i] = np.clip(final_polls[i] / 100.0, 0.0, 1.0)

    test_mask = years == TEST_YEAR
    return compute_metrics(y[test_mask], probs[test_mask])


def evaluate_lstm(hidden_size=128, seq_len_weeks=12):
    """LSTM is scored on the same test set by construction."""
    results = lstm_model.train_and_eval(
        hidden_size=hidden_size,
        seq_len_weeks=seq_len_weeks,
    )
    return {"accuracy": results["accuracy"], "brier": results["brier"]}


def evaluate_logreg():
    df = final_df.copy()
    df['incumbent_flag'] = (df['incumbency'] == 'incumbent').astype(int)
  
    features = [
        'poll_avg',
        'TTL_RECEIPTS',
        'TTL_INDIV_CONTRIB',
        'OTHER_POL_CMTE_CONTRIB',
        'POL_PTY_CONTRIB',
        'partisan',
        'is_midterm',
        'incumbent_flag'
    ]

    # Train/test split
    train = df[df['year'] < 2022]
    test = df[df['year'] == 2022]

    # Train logistic regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(train[features].fillna(0), train['won'])

    y_prob_logreg = logreg.predict_proba(test[features].fillna(0))[:, 1]

    pred_df = test[[
        'year','state','office','district','party','candidate_mit','won','incumbency','poll_avg']].copy()
    pred_df = pred_df.rename(columns={'candidate_mit': 'candidate'})
    pred_df['y_true'] = pred_df['won']
    pred_df['y_prob'] = y_prob_logreg
  
    filtered = filter_to_test_set(pred_df)
    return compute_metrics(filtered['y_true'],filtered['y_prob'])

def evaluate_gbm():
    df = final_df.copy()
    df['incumbent_flag'] = (df['incumbency'] == 'incumbent').astype(int)

    features = [
      'poll_avg',
      'TTL_RECEIPTS',
      'TTL_INDIV_CONTRIB',
      'OTHER_POL_CMTE_CONTRIB',
      'POL_PTY_CONTRIB',
      'partisan',
      'is_midterm',
      'incumbent_flag'
    ]
  
    #train test split
    train = df[df['year'] < 2022]
    test = df[df['year'] == 2022]

    #train gradient boosting
    gbm = GradientBoostingClassifier()
    gbm.fit(train[features].fillna(0), train['won'])
  
    y_prob_gbm = gbm.predict_proba(test[features].fillna(0))[:, 1]
  
    pred_df = test[[
        'year','state','office','district','party','candidate_mit','won','incumbency','poll_avg']].copy()
    pred_df = pred_df.rename(columns={'candidate_mit': 'candidate'})
    pred_df['y_true'] = pred_df['won']
    pred_df['y_prob'] = y_prob_gbm
  
    filtered = filter_to_test_set(pred_df)
    return compute_metrics(filtered['y_true'],filtered['y_prob'])

FEATURE_SETS = {
    "poll_only": ['poll_avg'],
    "poll_plus_basic": [
        'poll_avg',
        'incumbent_flag',
        'partisan',
        'is_midterm'],
    "all_features": [
        'poll_avg',
        'incumbent_flag',
        'partisan',
        'is_midterm',
        'TTL_RECEIPTS',
        'TTL_INDIV_CONTRIB',
        'OTHER_POL_CMTE_CONTRIB',
        'POL_PTY_CONTRIB']
}

def evaluate_logreg_t1(feature_set):
    df = final_df.copy()
    df['incumbent_flag'] = (df['incumbency'] == 'incumbent').astype(int)

    features = FEATURE_SETS[feature_set]

    train = df[df['year'] < 2022]
    test = df[df['year'] == 2022]

    model = LogisticRegression(max_iter = 1000)
    model.fit(train[features].fillna(0), train['won'])

    y_prob = model.predict_proba(test[features].fillna(0))[:, 1]

    return compute_metrics(test['won'], y_prob)

def evaluate_gbm_t1(feature_set):
    df = final_df.copy()
    df['incumbent_flag'] = (df['incumbency'] == 'incumbent').astype(int)

    features = FEATURE_SETS[feature_set]

    train = df[df['year'] < 2022]
    test = df[df['year'] == 2022]

    model = GradientBoostingClassifier()
    model.fit(train[features].fillna(0), train['won'])

    y_prob = model.predict_proba(test[features].fillna(0))[:, 1]

    return compute_metrics(test['won'], y_prob)
    
# Used Claude to format the printing again here and below
def pretty_print(name, result):
    if result is None:
        print(f"  {name:<45} [not yet implemented]")
    else:
        print(f"  {name:<45} acc = {result['accuracy']:.3f}   "
              f"brier = {result['brier']:.4f}")


if __name__ == "__main__":
    n_test = len(load_test_candidates())
    print("=" * 75)
    print(f"Evaluation on {TEST_YEAR} test set ({n_test} candidates)")
    print("All models scored on identical candidate set from test_candidates.csv")
    print("=" * 75)
    print()

    pretty_print("Naive poll baseline (poll leader wins)", naive_poll_baseline())
    pretty_print("LSTM (hidden=128, seq=12 weeks)", evaluate_lstm())
    pretty_print("Logistic regression", evaluate_logreg())
    pretty_print("Gradient boosted trees", evaluate_gbm())

    print()
    print("=" * 75)
    print("Experiment 1: Feature Set Variation")
    print("=" * 75)

    for name in FEATURE_SETS:
        print()
        pretty_print(f"LogReg ({name})", evaluate_logreg_t1(name))
        pretty_print(f"GBM ({name})", evaluate_gbm_t1(name))

""" FILE DESCRIPTION
Create an evaluation script to train and evaluate the models for our two tests in an easy to read, consistent way.
All models are scored on the same 2022 candidates that survived the filters.
(build_lstm_features.py's filters (candidates with FEC matches +
enough polling history)). 

Models:
  Naive poll baseline: predict winner = candidate with higher final poll.
  Logistic regression
  Gradient boosted trees
  LSTM
"""
