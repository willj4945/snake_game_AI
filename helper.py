import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def close():
    try:
        plt.close('all')
    except Exception:
        # Tk raises TclError if its window was already torn down. Forget the figures
        # so matplotlib's exit hook doesn't try to destroy them again.
        Gcf.figs.clear()
