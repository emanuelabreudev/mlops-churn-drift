import pandas as pd

from churn_mlops.data.preprocess import clean, split


def test_totalcharges_blank_becomes_zero(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype == "float64"
    assert not df["TotalCharges"].isna().any()
    novos = raw_df["tenure"] == 0
    assert (df.loc[novos.values, "TotalCharges"] == 0.0).all()


def test_id_dropped_and_target_binary(raw_df):
    df = clean(raw_df)
    assert "customerID" not in df.columns
    assert set(df["Churn"].unique()) <= {0, 1}


def test_senior_citizen_is_categorical(raw_df):
    df = clean(raw_df)
    assert set(df["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_split_stratified_no_overlap(clean_df):
    train, val, test = split(clean_df, "Churn", test_size=0.15, val_size=0.15, seed=42)
    n = len(clean_df)
    assert len(train) + len(val) + len(test) == n
    assert abs(len(test) / n - 0.15) < 0.02
    overall = clean_df["Churn"].mean()
    for part in (train, val, test):
        assert abs(part["Churn"].mean() - overall) < 0.05


def test_split_deterministic(clean_df):
    a = split(clean_df, "Churn", 0.15, 0.15, seed=42)[0]
    b = split(clean_df, "Churn", 0.15, 0.15, seed=42)[0]
    pd.testing.assert_frame_equal(a, b)
