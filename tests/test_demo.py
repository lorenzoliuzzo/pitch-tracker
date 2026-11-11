from pitch_tracker.demo import WEAK_FUNDAMENTAL, error_table, format_table


def test_weak_fundamental_table_shows_what_each_estimator_does() -> None:
    table = error_table(WEAK_FUNDAMENTAL)
    assert set(table) == {"E2", "A2", "D3", "G3", "B3", "E4"}
    for row in table.values():
        assert row["hps"] is not None and abs(row["hps"]) < 3.0
        assert row["autocorr"] is not None and abs(row["autocorr"]) < 2.0
        assert row["peak"] is not None and row["peak"] > 1100.0


def test_table_prints_one_line_per_string() -> None:
    text = format_table(error_table([1.0]))
    assert len(text.splitlines()) == 7
