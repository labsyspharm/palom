"""Small matplotlib helpers shared by the library and CLI QC plots."""
import matplotlib.pyplot as plt


def set_subplot_size(w, h, ax=None):
    """Resize the figure so that `ax` itself is w x h inches."""
    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    figw = float(w) / (right - left)
    figh = float(h) / (top - bottom)
    ax.figure.set_size_inches(figw, figh)
