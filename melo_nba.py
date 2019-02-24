#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from skopt import gp_minimize

from melo import Melo
from nba_games import games


dates = games['date']
labels1 = games['home_team']
labels2 = games['away_team']
spreads = games['home_points'] - games['away_points']
totals = games['home_points'] + games['away_points']


def melo_wrapper(mode, k, bias, regress, smooth, verbose=False):
    """
    Wrapper to pass arguments to the Melo library.

    """
    values, lines = {
        'minus': (spreads, np.arange(-60.5, 61.5)),
        'plus': (totals, np.arange(-115.5, 300.5)),
    }[mode]

    return Melo(
        dates, labels1, labels2, values, mode,
        lines=lines, k=k, bias=bias, smooth=smooth,
        regress=lambda t: regress if t > np.timedelta64(12, 'W') else 0
    )


def from_cache(mode, retrain=False, **kwargs):
    """
    Load the melo args from the cache if available, otherwise
    train and cache a new instance.

    """
    cachefile = Path('cachedir', '{}.cache'.format(mode.lower()))

    if not retrain and cachefile.exists():
        args = np.loadtxt(cachefile)
        return melo_wrapper(mode, *args)

    def obj(args):
        melo = melo_wrapper(mode, *args)
        return melo.entropy()

    bounds = {
        'minus': [
            (0.0,    0.3),
            (0.0,    0.3),
            (0.0,    0.5),
            (0.0,   15.0),
        ],
        'plus': [
            (0.0,    0.3),
            (-0.01, 0.01),
            (0.0,    0.5),
            (0.0,   15.0),
        ],
    }[mode]

    res = gp_minimize(obj, bounds, n_calls=100, n_jobs=4, verbose=True)

    print("mode: {}".format(mode))
    print("best mean absolute error: {:.4f}".format(res.fun))
    print("best parameters: {}".format(res.x))

    if not cachefile.parent.exists():
        cachefile.parent.mkdir()

    np.savetxt(cachefile, res.x)
    return melo_wrapper(mode, *res.x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='calibrate model parameters for point spreads and totals',
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--retrain', action='store_true', default=False,
        help='retrain even if model args are cached'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for mode in 'minus', 'plus':
        from_cache(mode, **kwargs)
else:
    nba_spreads = from_cache('minus')
    nba_totals = from_cache('plus')
