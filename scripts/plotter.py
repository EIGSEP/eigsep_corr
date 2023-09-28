import time
import matplotlib.pyplot as plt
import numpy as np


# XXX add option to make multiple figures for plotting cross mag and phase
def _plot_init(y, x=None, labels=None):
    """
    Live plot initializer.

    Parameters
    ----------
    y : array-like
        Data to plot. Shape (n,) or (m, n). Rows are interpreted as multiple
        data sets.
    x : array-like
        The x-axis. If None, defaults to np.arange(len(y)).
    labels : list of str
        Labels for each data set.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    lines : list of matplotlib.lines.Line2D
        The line objects.

    """
    y = np.atleast_2d(y)
    if x is None:
        x = np.arange(y.shape[1])
    if labels is None:
        labels = [""] * y.shape[0]
        use_legend = False
    else:
        use_legend = True
    lines = []
    plt.ion()
    fig = plt.figure()
    for i in range(y.shape[0]):
        (line,) = plt.plot(x, y[i], label=labels[i])
        lines.append(line)
    # plt.ylim()
    if use_legend:
        plt.legend()
    return fig, lines


def plot_auto(fpga, auto):
    """
    Live plotting of correlation output from FPGA

    Parameters
    ----------
    fpga: eigsep_corr.fpga.EigsepFpga
        Instance of EigsepFpga object.
    auto : int or list of ints
        Auto-correlations to plot.
    """
    if isinstance(auto, int):
        auto = [auto]

    # freq = np.linspace(0, 250, 2048) #get sample rate from fpga, fftshift?
    freq = np.arange(2048)
    y = np.zeros((len(auto), 2048))
    for i in auto:
        y[i] = fpga.read_auto(i=i)
    labels = ["auto {}".format(i) for i in auto]
    fig, lines = _plot_init(y, x=freq, labels=labels)

    try:
        while True:
            for i in auto:
                x = fpga.read_auto(i=i)
                lines[i].set_ydata(x)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
    except KeyboardInterrupt:
        plt.close(fig)
        print("Plotting stopped.")


def plot_cross(fpga, cross):
    if isinstance(cross, str):
        cross = [cross]

    freq = np.arange(2048)
    y = np.zeros((len(cross), 2048))
    for ij in cross:
        x = fpga.read_cross(ij=ij)
        real = x[::2]
        imag = x[1::2]
        mag = np.sqrt(real**2 + imag**2)
        # phase = np.arctan2(imag, real) #XXX
        y[ij] = mag
    labels = ["cross {}".format(i) for i in cross]
    fig, lines = _plot_init(y, x=freq, labels=labels)
    try:
        while True:
            for ij in cross:
                x = fpga.read_cross(ij=ij)
                real = x[::2]
                imag = x[1::2]
                mag = np.sqrt(real**2 + imag**2)
                lines[ij].set_ydata(mag)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
    except KeyboardInterrupt:
        plt.close(fig)
        print("Plotting stopped.")
