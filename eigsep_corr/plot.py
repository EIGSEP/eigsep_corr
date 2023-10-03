import time
import matplotlib.pyplot as plt
import numpy as np

NCHAN = 1024
SAMPLE_RATE = 500


def plot(
    redis,
    pairs=["0", "1", "2", "3", "4", "5", "02", "04", "24", "13", "15", "35"],
    x=np.linspace(0, SAMPLE_RATE / 2, num=NCHAN, endpoint=False),
    plot_delay=False,
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
        colors[p] = f"C{i}"
    mag_lines = {}
    phase_lines = {}
    if plot_delay:
        dly_lines = {}
        nrows = 3
    else:
        nrows = 2
    plt.ion()
    fig, axs = plt.subplots(figsize=(10, 10), nrows=nrows)
    axs[0].sharex(axs[1])
    axs[0].set_ylabel("Magnitude")
    axs[1].set_ylabel("Phase")
    axs[1].set_xlabel("Frequency (MHz)")
    if plot_delay:
        axs[2].set_ylabel("Delay spectrum")
        axs[2].set_xlabel("Delay (ns)")
    for p in pairs:
        line_kwargs = {"color": colors[p], "label": p}
        (line,) = axs[0].semilogy(x, np.ones(NCHAN), **line_kwargs)
        mag_lines[p] = line
        if len(p) == 2:
            (line,) = axs[1].plot(x, np.zeros(NCHAN), **line_kwargs)
            phase_lines[p] = line
            if plot_delay:
                N_dlys = NCHAN // 2 + 1
                tau = np.arange(N_dlys) / (SAMPLE_RATE * 1e-3)
                (line,) = axs[2].plot(tau, np.ones(N_dlys), **line_kwargs)
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
                if len(p) == 1:  # auto
                    mag_lines[p].set_ydata(data)
                    ymax_mag = np.maximum(ymax_mag, data.max())
                    axs[0].set_ylim(0, ymax_mag)
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

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)

    except KeyboardInterrupt:
        plt.close(fig)
        print("Plotting stopped.")
