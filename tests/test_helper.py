import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf

import helper


def test_close_tolerates_already_destroyed_window(monkeypatch):
    # Regression: Tk raised TclError on quit when its window was already gone.
    plt.figure()

    def broken_close(which):
        raise RuntimeError('application has been destroyed')
    monkeypatch.setattr(helper.plt, 'close', broken_close)

    helper.close()

    assert not Gcf.figs  # so matplotlib's atexit hook won't retry and print a traceback
