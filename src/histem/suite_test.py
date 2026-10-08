from histem.suite import Scores, Suite


def _scores(**fit: float) -> Scores:
    f = {(k, "c"): v for k, v in fit.items()}
    return Scores(fit=f, description_length=0.0, objective=sum(fit.values()))


def test_regressions_flags_only_datasets_beyond_tolerance() -> None:
    before = _scores(a=1.0, b=1.0)
    after = _scores(a=1.04, b=1.2)
    assert Suite.regressions(before, after, tolerance=0.05) == ["b"]
