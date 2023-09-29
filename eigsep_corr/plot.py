import time
import matplotlib.pyplot as plt
import numpy as np

CHANS = 1024

# XXX add option to make multiple figures for plotting cross mag and phase
def _plot_init(y, x=None, labels=None, ylim=None):
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
    ylim : tup
        Limit on y-axis. Gets passed to plt.ylim.

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
    if ylim is not None:
        plt.ylim(*ylim)
    if use_legend:
        plt.legend()
    return fig, lines


def plot(fpga, auto, x=np.arange(CHANS)):
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

    labels = ["auto {}".format(i) for i in auto]
    
    y = np.zeros((len(auto), CHANS))
    for i in auto:
        y[i] = fpga.read_auto(i=i)[:CHANS]
    fig, lines = _plot_init(y, labels=labels, x=x)

    try:
        while True:
            for i in auto:
                x = fpga.read_auto(i=i)
                s = x[:CHANS] + x[CHANS:]
                #d = x[:CHANS] - x[CHANS:]
                lines[i].set_ydata(s)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
    except KeyboardInterrupt:
        plt.close(fig)
        print("Plotting stopped.")


def plot(fpga, pairs=["0", "1", "2", "3", "4", "5", "02", "04", "24", "13", "15", "35"], x=np.arange(CHANS)):
    
    if isinstance(pairs, str):
        pairs = [pairs]

    y = np.zeros((len(pairs), CHANS))
    for ij in pairs:
        if len(ij) == 1:
            data = fpga.read_auto(i=int(ij))
        else:
            data = fpga.read_cross(ij=ij)
            real = data[::2]
            imag = data[1::2]
        mag = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
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
