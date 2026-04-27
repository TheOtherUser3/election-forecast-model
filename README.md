# Predicting U.S. House and Senate Election Outcomes

DS340 final project, Spring 2026. Catherine Archambault and Dawson Maska, Boston University.

We trained an LSTM to predict winners of U.S. House and Senate general elections from 12-week sequences of weekly polling averages plus eight static features (fundraising, incumbency, partisan lean, midterm flag), and compared it against logistic regression, gradient boosted trees, and a naive poll-leader baseline on the 2022 midterms. The LSTM won on both accuracy (79.7%) and Brier score (0.141). Full writeup is in the paper.

## Repo layout

```
process_data.py          # Cleans MIT election results and FEC fundraising,
                         # fuzzy-matches them, writes elections_clean.csv
process_polls.py         # Joins FiveThirtyEight polls onto the elections data
                         # and computes partisan lean, writes
                         # elections_clean_with_polls.csv, merged into process_data.py for ease of use
build_lstm_features.py   # Builds per-candidate weekly polling sequences and
                         # the static feature matrix, writes lstm_data.npz
                         # and test_candidates.csv
lstm_model.py            # LSTM architecture, training loop, and the
                         # hidden-size and sequence-length sweeps
baseline_tests           # Logistic Regression and Gradient-Boosted Trees architectures
evaluate.py              # Scores all four models on the same 133-candidate
                         # 2022 test set for both Test 1 and Test 2
```

## How to run

You need Python 3.10 or newer and the following packages:

```
pandas
numpy
scikit-learn
torch
rapidfuzz
```

Download the code. Then run the pipeline in order:

```
python process_data.py
python build_lstm_features.py
python evaluate.py
```

`process_data.py` is a little slow because of the fuzzy matching step. `build_lstm_features.py` and `evaluate.py` together run in well under a minute.  Altogether, only takes a couple minutes due to the small training set. `evaluate.py` will print accuracy and Brier scores for the naive baseline, the LSTM, logistic regression, and gradient boosting on the 2022 test set, followed by the Test 1 feature-set variation results.

To run just the LSTM ablations without the full evaluation, run `python lstm_model.py` directly. That triggers `run_all_sweeps()`, which tries hidden sizes 64, 128, and 256 at a 12-week sequence length, then sequence lengths of 4, 8, and 12 weeks at hidden size 128.

## Notes on the data

The effective training window is 1998 to 2020, with 2022 held out as the test set. The proposal originally said 1976 to 2020. We narrowed it because FiveThirtyEight's pre-1998 polling coverage is basically nonexistent. The fuzzy match between FEC and MIT records lands at about 58.7%, mostly due to pre-2000 FEC digital records being poor quality. Since most of these aren't included anyways due to the polling limitation, it is not a big issue. After requiring at least one observed poll in the 84 days before election day, the LSTM training set is 1,514 candidates and the test set is 133 candidates from 2022.

The 133-candidate test set is small by design. The LSTM needs polling to make a prediction, and the only races that get polled are competitive ones and the occasional safe race. That restricts our claims to, for the most part, competitive races, which happens to be the only set of races anyone actually wants a forecast for anyways.

## AI use

We used Claude AI for help with implementing the LSTM training loop, the class imbalance handling, fuzzy matching code, and various print formatting. We wrote the project design, feature engineering decisions, evaluation pipeline, and analysis ourselves. The data cleaning and modeling decisions are ours. Specific lines that came from AI assistance are flagged in inline comments throughout the code.

## Authors

Catherine Archambault handled the polling data preprocessing, the partisan lean computation, the logistic regression and gradient boosted tree baselines, and the Test 1 feature-set variation experiment.

Dawson Maska handled the MIT and FEC data cleaning and joining, the LSTM input construction, the LSTM model architecture and training, the naive baseline, and the Test 2 model comparison and ablations.

The paper submitted on Blackboard was written collaboratively.
