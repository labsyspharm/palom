"""Small matplotlib helpers shared by the library and CLI QC plots."""
import matplotlib.pyplot as plt

# Displayed pixels per inch for a QC image panel. QC figures are written at 144
# dpi, so an image lands at half its pixel size -- big enough to read tissue
# structure, small enough that a whole-slide thumbnail is not a wall poster.
IMAGE_PX_PER_INCH = 288
# Inches reserved above the axes for the axes title plus a caller's suptitle.
TITLE_BAND_IN = 0.5


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


def size_axes_to_image(ax, title_in=TITLE_BAND_IN, min_w_in=None):
    """Size `ax` to the image it displays and reserve a title band above it.

    The single sizing treatment for every single-panel image QC figure, applied
    where the figure is *drawn* rather than by whoever later saves it. Sizing
    from the caller only ever reaches the figures that caller made itself: it is
    why the per-object coarse-alignment plots came out at matplotlib's default
    6.4 x 4.8 while the whole-slide one, drawn by the same code, was sized.

    Two details that were each learned once and then not shared:

    - the band is made by *growing* the figure, not by shrinking the axes into
      it. Shrinking fails outright (`bottom cannot be >= top`) when the image is
      less than `title_in` tall, which a small object's coarse match easily is.
    - `min_w_in` floors the width for plots whose titles and colorbars carry
      more text than a small image can sit under.
    """
    h_px, w_px = ax.images[0].get_array().shape[:2]
    w_in, h_in = w_px / IMAGE_PX_PER_INCH, h_px / IMAGE_PX_PER_INCH
    if min_w_in is not None and w_in < min_w_in:
        w_in, h_in = min_w_in, h_in * min_w_in / w_in
    set_subplot_size(w_in, h_in, ax=ax)
    # hang the axes from the top of its slot so it stays under its title
    ax.set_anchor("N")
    fig = ax.figure
    figw, figh = fig.get_size_inches()
    fig.set_size_inches(figw, figh + title_in)
    fig.subplots_adjust(top=1 - title_in / (figh + title_in))
    return fig
