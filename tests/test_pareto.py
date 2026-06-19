"""Tests for the dependency-free non-dominated sort and Pareto front."""

from __future__ import annotations

import numpy as np

from robotactuatormdo.studies.pareto import non_dominated_indices


def test_known_2d_front():
    # minimize both columns
    pts = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
    assert set(non_dominated_indices(pts)) == {0, 1, 2}  # (2,3) dominated by (2,2)


def test_duplicates_all_kept():
    pts = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert set(non_dominated_indices(pts)) == {0, 1}


def test_single_point():
    assert non_dominated_indices(np.array([[5.0, 5.0]])) == [0]


def test_fully_dominated_chain():
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert non_dominated_indices(pts) == [0]


def test_three_objectives():
    pts = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    # row2 dominated by row1 (1,2,3) ? 1<=3,2<=3,3<=3 and some < -> yes dominated.
    # row0 vs row1: neither dominates (1<2 but 3>2). both on front.
    assert set(non_dominated_indices(pts)) == {0, 1}
