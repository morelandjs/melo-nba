#!/usr/bin/env python3

from datetime import datetime, timedelta

import numpy as np
from pyDOE import lhs

from melo import Melo
from nba_games import games


dates = games['date']
labels1 = games['home_team']
labels2 = games['away_team']
spreads = games['home_points'] - games['away_points']
totals = games['home_points'] + games['away_points']


def melo_wrapper(mode, k, bias, decay, smooth):
    """
    Thin wrapper to pass arguments to the Melo library.

    """
    values, lines = {
        'Fermi': (spreads, np.arange(-60.5, 61.5)),
        'Bose': (totals, np.arange(-115.5, 300.5)),
    }[mode]

    return Melo(
        dates, labels1, labels2, values, lines=lines,
        mode=mode, k=k, bias=bias, smooth=smooth,
        decay=lambda t: 1 if t < timedelta(weeks=20) else decay
    )


def calibrate_model(mode, bounds, points=50):
    """
    Calibrate model hyperparameters

    """
    lhsmin, lhsmax = map(np.array, zip(*bounds))
    X = lhsmin + (lhsmax - lhsmin) * lhs(4, samples=points, criterion='maximin')
    y = np.zeros((points, 1))

    for p, args in enumerate(X):
        print('[INFO][melo_nba] evaluating design point: {}'.format(p))
        melo = melo_wrapper(mode, *args)
        y[p] = np.std(melo.residuals(statistic='mean'))

    return X[np.argmin(y)]


if __name__ == '__main__':

    for mode in 'Fermi', 'Bose':

        bounds = {
            'Fermi': [(0.09, 0.15), (0.19, 0.25), (0.58, 0.68), (9.0, 9.4)],
            'Bose': [(0.0, 0.3), (-1e-3, 1e-3), (0.5, 1.0), (0.0, 9.4)],
        }[mode]

        args = calibrate_model('Fermi', bounds)

        print('[INFO][melo_nba] mode={}:'.format(mode),
            'k={:.2f}, bias={:.2f}, decay={:.2f}, smooth={:.2f}'.format(*args))

        nba_spreads = melo_wrapper(mode, *args)
        residuals = nba_spreads.residuals(statistic='mean')
        mae = np.abs(residuals).mean()
        std = np.std(residuals)
        print('[INFO][melo_nba] std dev={:.2f}, mae={:.2f}'.format(std, mae))

else:
   nba_spreads = melo_wrapper('Fermi', 0.13, 0.23, 0.65, 9.18)
   nba_totals = melo_wrapper('Bose', 0.12,  0.0, 0.68, 5.02)
