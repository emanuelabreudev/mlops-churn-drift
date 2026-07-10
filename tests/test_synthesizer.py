import numpy as np

from churn_mlops.drift.scenario import INTENSITIES, apply_covariate_drift, shift_odds
from churn_mlops.drift.synthesizer import GaussianCopulaSynthesizer


def test_synthesizer_preserves_schema_and_ranges(clean_df, schema):
    synth = GaussianCopulaSynthesizer(schema, seed=42).fit(clean_df)
    sample = synth.sample(300, seed=7)
    assert list(sample.columns) == schema["numeric_features"] + schema["categorical_features"]
    assert len(sample) == 300
    assert not sample.isna().any().any()
    for col in schema["categorical_features"]:
        assert set(sample[col].unique()) <= set(schema["categories"][col])
    assert sample["tenure"].min() >= 0
    assert sample["MonthlyCharges"].between(10, 130).all()


def test_synthesizer_is_deterministic_per_seed(clean_df, schema):
    synth = GaussianCopulaSynthesizer(schema, seed=42).fit(clean_df)
    a, b = synth.sample(100, seed=3), synth.sample(100, seed=3)
    assert a.equals(b)
    assert not a.equals(synth.sample(100, seed=4))


def test_synthesizer_preserves_correlation_sign(clean_df, schema):
    # tenure e TotalCharges são fortemente correlacionados; a copula deve manter isso
    synth = GaussianCopulaSynthesizer(schema, seed=42).fit(clean_df)
    sample = synth.sample(1500, seed=5)
    assert sample["tenure"].corr(sample["TotalCharges"]) > 0.5


def test_covariate_drift_shifts_distributions(clean_df, schema):
    synth = GaussianCopulaSynthesizer(schema, seed=42).fit(clean_df)
    base = synth.sample(1500, seed=9)
    rng = np.random.default_rng(0)
    drifted = apply_covariate_drift(base, INTENSITIES["moderate"], rng)

    assert drifted["MonthlyCharges"].mean() > base["MonthlyCharges"].mean() * 1.15
    assert drifted["tenure"].mean() < base["tenure"].mean()
    assert (drifted["Contract"] == "Month-to-month").mean() > (base["Contract"] == "Month-to-month").mean()
    # coerência: quem ganhou fibra não pode manter add-ons "No internet service"
    fiber = drifted["InternetService"] == "Fiber optic"
    assert not (drifted.loc[fiber, "OnlineSecurity"] == "No internet service").any()


def test_sampled_data_is_structurally_consistent(clean_df, schema):
    synth = GaussianCopulaSynthesizer(schema, seed=42).fit(clean_df)
    sample = synth.sample(1000, seed=11)
    no_internet = sample["InternetService"] == "No"
    assert (sample.loc[no_internet, "OnlineSecurity"] == "No internet service").all()
    assert not (sample.loc[~no_internet, "StreamingTV"] == "No internet service").any()
    no_phone = sample["PhoneService"] == "No"
    assert (sample.loc[no_phone, "MultipleLines"] == "No phone service").all()
    assert not (sample.loc[~no_phone, "MultipleLines"] == "No phone service").any()


def test_shift_odds_increases_probability():
    p = np.array([0.1, 0.3, 0.5])
    shifted = shift_odds(p, 1.6)
    assert (shifted > p).all()
    assert (shifted < 1).all()
    np.testing.assert_allclose(shift_odds(p, 1.0), p)
