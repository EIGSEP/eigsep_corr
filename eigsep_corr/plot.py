import time
import matplotlib.pyplot as plt
import numpy as np

NCHAN = 1024


def plot(
    redis,
    pairs=["0", "1", "2", "3", "4", "5", "02", "04", "24", "13", "15", "35"],
    x=np.arange(NCHAN),
    ylim_mag=None,
    ylim_phase=None,
    sleep=0.1,
):
    """
    Live plotting of correlation output from FPGA

    Parameters
    ----------
    redis: redis.Redis
        Instance of redis.Redis object where data is stored.
    pairs : str or list of str
        Correlation pairs to plot. Defaults to all pairs.
    x : array-like
        The x-axis. If None, defaults to np.arange(len(y)).
    ylim_mag : tup
        Limit on y-axis for magnitude. Gets passed to plt.ylim.
    ylim_phase : tup
        Limit on y-axis for phase. Gets passed to plt.ylim.
    sleep : float
        Time (in seconds) to sleep between updates.

    """
    if isinstance(pairs, str):
        pairs = [pairs]

    mag_lines = {}
    phase_lines = {}
    plt.ion()
    fig, axs = plt.subplots(figsize=(10, 10), ncols=2, sharex=True)
    for p in pairs:
        (line,) = axs[0].plot(x, np.zeros(NCHAN), label=p)
        mag_lines[p] = line
        if len(p) == 2:
            (line,) = axs[1].plot(x, np.zeros(NCHAN), label=p)
            phase_lines[p] = line
    if ylim_mag is not None:
        axs[0].set_ylim(*ylim_mag)
    if ylim_phase is not None:
        axs[1].set_ylim(*ylim_phase)
    axs[0].legend()
    axs[1].legend()

    try:
        while True:
            for p in pairs:
                data = redis.get(f"data:{p}")
                if len(p) == 1:  # auto
                    mag_lines[p].set_ydata(data)
                else:  # cross
                    real = data[::2]
                    imag = data[1::2]
                    mag = np.sqrt(real**2 + imag**2)
                    phase = np.arctan2(imag, real)
                    mag_lines[p].set_ydata(mag)
                    phase_lines[p].set_ydata(phase)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)

    except KeyboardInterrupt:
        plt.close(fig)
        print("Plotting stopped.")
