import time
import matplotlib.pyplot as plt
import numpy as np

NCHAN = 1024
SAMPLE_RATE = 500


def plot_live(
    redis,
    pairs=["0", "1", "2", "3", "4", "5", "02", "04", "24", "13", "15", "35"],
    x=np.linspace(0, SAMPLE_RATE / 2, num=NCHAN, endpoint=False),
    plot_delay=False,
    sleep=0.1,
    log_scale=True,
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
        The x-axis. Defaults to the frequency channels defined by NCHAN and
        SAMPLE_RATE.
    plot_delay : bool
        Whether to plot delay spectrum, i.e., the Fourier transform of the
        correlation.
    sleep : float
        Time (in seconds) to sleep between updates.

    """
    if isinstance(pairs, str):
        pairs = [pairs]

    colors = {}
    for i, p in enumerate(pairs):
        # there are only 10 colors in the default color cycle
        if i == 0:
            colors[p] = "black"
        elif i == 1:
            colors[p] = "lime"
        else:  # pairs 0-9, and repeats if more than 12 pairs
            colors[p] = f"C{i-2}"

    mag_lines = {}
    phase_lines = {}
    if plot_delay:
        dly_lines = {}
        nrows = 3
    else:
        nrows = 2
    plt.ion()
    fig, axs = plt.subplots(figsize=(10, 10), nrows=nrows)
    axs[0].grid()
    axs[1].grid()
    axs[0].sharex(axs[1])
    axs[0].set_ylabel("Magnitude")
    axs[1].set_ylabel("Phase")
    axs[1].set_xlabel("Frequency (MHz)")
    if log_scale:
        axs[0].set_ylim(1e-2, 1e9)
    else:
        axs[0].set_ylim(0, 3e6)
    if plot_delay:
        axs[2].set_ylabel("Delay spectrum")
        axs[2].set_xlabel("Delay (ns)")
    for p in pairs:
        line_kwargs = {"color": colors[p], "label": p}
        if log_scale:
            (line,) = axs[0].semilogy(x, np.ones(NCHAN), **line_kwargs)
        else:
            (line,) = axs[0].plot(x, np.ones(NCHAN), **line_kwargs)
        mag_lines[p] = line
        if len(p) == 2:
            (line,) = axs[1].plot(x, np.zeros(NCHAN), **line_kwargs)
            phase_lines[p] = line
            if plot_delay:
                tau = np.fft.rfftfreq(NCHAN, d=x[1] - x[0])
                tau *= 1e3  # convert to ns
                (line,) = axs[2].plot(tau, np.ones_like(tau), **line_kwargs)
                dly_lines[p] = line
    ymax_mag = 0
    if plot_delay:
        ymax_dly = 0
    axs[1].set_ylim(-np.pi, np.pi)
    axs[0].legend(bbox_to_anchor=(1.1, 1.1), loc="upper right")
    try:
        while True:
            for p in pairs:
                dt = np.dtype(np.int32).newbyteorder(">")
                data = np.frombuffer(redis.get(f"data:{p}"), dtype=dt)
                cnt = redis.get("ACC_CNT")
                print(cnt)
                ymax_mag = np.maximum(ymax_mag, data.max())
                #axs[0].set_ylim(1e1, ymax_mag)
                if len(p) == 1:  # auto
                    mag_lines[p].set_ydata(data)
                else:  # cross
                    real = data[::2].astype(np.int64)
                    imag = data[1::2].astype(np.int64)
                    mag = np.sqrt(real**2 + imag**2)
                    phase = np.arctan2(imag, real)
                    mag_lines[p].set_ydata(mag)
                    phase_lines[p].set_ydata(phase)
                    if plot_delay:
                        dly = np.abs(np.fft.rfft(np.exp(1j * phase))) ** 2
                        dly_lines[p].set_ydata(dly)
                        ymax_dly = np.maximum(ymax_dly, dly.max())
                        axs[2].set_ylim(0, ymax_dly)
                        # assuming the peak is in 2nd Nyquist window
                        alias_peak = np.argmax(dly)
                        actual_peak = 2 * len(tau) - alias_peak
                        print(f"Delay in sample clocks: {actual_peak}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)

    except KeyboardInterrupt:
        plt.close(fig)
        print("Plotting stopped.")


def plot_from_file():
    raise NotImplementedError
