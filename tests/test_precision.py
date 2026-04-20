import math

import pytest

from chartpatterns.precision import smart_round_percent, smart_round_price


def test_smart_round_price_normal():
    # 6 sig figs, min 2 decimals. 123.456789 -> 123.457 but min keeps 2
    rounded = smart_round_price(123.456789)
    # magnitude=2, decimals = max(2, min(15, 6-1-2)) = 3
    assert rounded == 123.457


def test_smart_round_price_micro():
    # 1e-9 region: magnitude=-9, decimals = max(2, min(15, 6-1-(-9))) = 14
    rounded = smart_round_price(1.234e-9)
    assert rounded > 0
    assert rounded == pytest.approx(1.234e-9, rel=1e-3)


def test_smart_round_price_zero():
    assert smart_round_price(0.0) == 0.0


def test_smart_round_price_nan_returns_zero():
    assert smart_round_price(math.nan) == 0.0


def test_smart_round_price_inf_returns_zero():
    assert smart_round_price(math.inf) == 0.0
    assert smart_round_price(-math.inf) == 0.0


def test_smart_round_percent_basic():
    # 4 sig figs, min 4 decimals
    rounded = smart_round_percent(1.23456)
    # magnitude=0, decimals = max(4, min(15, 4-1-0)) = 4
    assert rounded == 1.2346


def test_smart_round_percent_nan_returns_zero():
    assert smart_round_percent(math.nan) == 0.0
