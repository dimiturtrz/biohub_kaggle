import math

from celltrack.eval.official_ruler import MetricCounts, OfficialRuler, TrackingScore


def _score(edges: tuple[int, int, int], divisions: tuple[int, int, int] = (0, 0, 0)) -> TrackingScore:
    return TrackingScore(edges=MetricCounts(*edges), divisions=MetricCounts(*divisions))


def test_jaccard():
    assert MetricCounts(tp=6, fp=2, fn=2).jaccard() == 0.6


def test_jaccard_is_nan_on_an_empty_axis():
    # The donor's own convention: an axis nothing was proposed for scores NaN, not 0.0, so a movie
    # without divisions cannot drag the division term down.
    assert math.isnan(MetricCounts(tp=0, fp=0, fn=0).jaccard())


def test_is_empty():
    assert MetricCounts(0, 0, 0).is_empty()
    assert not MetricCounts(0, 0, 1).is_empty()


def test_metric_counts_add():
    assert MetricCounts(1, 2, 3) + MetricCounts(10, 20, 30) == MetricCounts(11, 22, 33)


def test_micro():
    # Summed counts then ONE ratio: 1/2 and 99/100 micro to 100/102, not to the 0.745 mean of the two.
    micro = TrackingScore.micro([_score((1, 1, 0)), _score((99, 1, 0))])
    assert micro.edges == MetricCounts(tp=100, fp=2, fn=0)
    assert micro.edges.jaccard() == 100 / 102


def test_micro_of_nothing_is_empty():
    assert TrackingScore.micro([]).edges.is_empty()


def test_value():
    assert _score((6, 2, 2), (2, 0, 2)).value() == 0.6 + 0.1 * 0.5


def test_value_drops_the_division_term_when_no_movie_has_a_division():
    assert _score((6, 2, 2)).value() == 0.6


def test_rows():
    ruler = OfficialRuler(movies=("a", "b"), score_movie=lambda _: _score((1, 0, 0)))
    assert [stem for stem, _ in ruler.rows()] == ["a", "b"]


def test_rows_skips_a_movie_with_no_prediction():
    scored = {"a": _score((1, 0, 0))}
    ruler = OfficialRuler(movies=("a", "b"), score_movie=scored.get)
    assert [stem for stem, _ in ruler.rows()] == ["a"]
