'Adapted from https://github.com/limix/limix-plot/blob/master/limix_plot/_qqplot.py'

def get_pyplot():
    """
    Get :mod:`matplotlib.pyplot`.

    Returns
    -------
    pyplot : :mod:`matplotlib.pyplot`
        MATLAB-like interface.
    """
    from sys import platform as sys_pf
    from matplotlib import rcParams

    if get_pyplot.pyplot is not None:
        return get_pyplot.pyplot

    if "backend" not in rcParams and sys_pf == "darwin":
        from matplotlib import use as _backend_use

        _backend_use("TkAgg")

    from matplotlib import pyplot

    get_pyplot.pyplot = pyplot
    return pyplot

get_pyplot.pyplot = None

def qqplot(
    a,
    label=None,
    alpha=0.05,
    cutoff=0.1,
    line=True,
    pts_kws=None,
    band_kws=None,
    ax=None,
    show_lambda=True,
    lambda_in_label=False,  
    ):
    """
    Quantile-Quantile plot of observed p-values versus theoretical ones.
    
    Returns
    -------
    lamb : float
        Genomic inflation factor (lambda).
    """
    from numpy import asarray, sort, log10, arange

    plt = get_pyplot()

    a = asarray(a)
    if a.ndim > 1:
        a = a.squeeze()

    if ax is None:
        ax = plt.gca()

    if pts_kws is None:
        pts_kws = dict()
    if "marker" not in pts_kws:
        pts_kws["marker"] = "o"
    if "linestyle" not in pts_kws:
        pts_kws["linestyle"] = ""
    if "markeredgecolor" not in pts_kws:
        pts_kws["markeredgecolor"] = None

    if band_kws is None:
        band_kws = dict()
    if "facecolor" not in band_kws:
        band_kws["facecolor"] = "#DDDDDD"
    if "linewidth" not in band_kws:
        band_kws["linewidth"] = 0
    if "zorder" not in band_kws:
        band_kws["zorder"] = -1
    if "alpha" not in band_kws:
        band_kws["alpha"] = 1.0

    pv = sort(a)
    ok = _subsample(pv, cutoff)

    qnull = -log10((0.5 + arange(len(pv))) / len(pv))
    qemp = -log10(pv)

    lamb = _compute_lambda(pv)

    if label is not None:
        if lambda_in_label:
            pts_kws["label"] = f"{label} (λ={lamb:.3f})"
        else:
            pts_kws["label"] = label
    elif lambda_in_label:
        pts_kws["label"] = f"λ={lamb:.3f}"

    ax.plot(qnull[ok], qemp[ok], **pts_kws)

    qmax = max(qnull[ok].max(), qemp[ok].max())

    xmin = qnull[ok].min()
    xmax = qnull[ok].max()

    if line:
        ax.plot([xmin, xmax], [xmin, xmax], color="black", zorder=0, linestyle="--")

    if alpha is not None:
        _plot_confidence_band(ok, qnull, alpha, ax, qmax, band_kws)

    if show_lambda and not lambda_in_label:
        _plot_lambda_text(lamb, ax, pts_kws)
        _adjust_lambda_texts(ax)

    ax.set_ylabel("-log$_{10}$pv observed", fontsize=14)
    ax.set_xlabel("-log$_{10}$pv expected", fontsize=14)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    return lamb  


def _compute_lambda(pv):
    """Compute genomic inflation factor (lambda)."""
    from numpy import median
    import scipy.stats as st

    chi2 = st.chi2(df=1)
    lamb = chi2.isf(median(pv)) / chi2.median()
    return lamb


def _plot_lambda_text(lamb, ax, pts_kws=None):
    """Plot lambda as text annotation."""
    # Set the color for the lambda text based on the points' color
    lambda_color = pts_kws.get('color') if pts_kws else 'black'

    # Define the lambda text
    text = "$\\lambda={:.3f}$".format(lamb)

    # Add text to the plot in the top-left corner
    ax.text(
        0.05, 0.95,
        text,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        color=lambda_color,
        fontsize=14,
        weight="bold"
    )


# Keep original _plot_lambda for backward compatibility
def _plot_lambda(pv, ax, pts_kws=None):
    lamb = _compute_lambda(pv)
    _plot_lambda_text(lamb, ax, pts_kws)


def _adjust_lambda_texts(ax):
    from adjustText import adjust_text

    texts = []
    for t in ax.texts:

        if "$\\lambda" in t.get_text():
            texts.append(t)

    if len(texts) > 1:
        y = texts[0].get_position()[1]
        for i, t in enumerate(texts[1:]):
            xy = t.get_position()
            t.set_position((xy[0], y - (i + 1) * 0.05))
            #t.set_color(lambda_color)  # Set color for each lambda text

        adjust_text(
            texts, autoalign="y", only_move={"text": "y"}, text_from_points=False
        )    

def _rank_confidence_band(nranks, significance_level, ok):
    from numpy import arange, flipud, ascontiguousarray
    from scipy.special import betaincinv

    alpha = significance_level

    k0 = arange(1, nranks + 1)
    k1 = flipud(k0).copy()

    k0 = ascontiguousarray(k0[ok])
    k1 = ascontiguousarray(k1[ok])

    my_ok = k1 / k0 / (k1[0] / k0[0]) > 1e-4
    k0 = ascontiguousarray(k0[my_ok])
    k1 = ascontiguousarray(k1[my_ok])

    top = betaincinv(k0, k1, 1 - alpha)
    bottom = betaincinv(k0, k1, alpha)

    return (my_ok, bottom, top)


def _plot_confidence_band(ok, null_qvals, significance_level, ax, qmax, band_kws):
    from numpy import log10

    (cb_ok, bo, to) = _rank_confidence_band(len(null_qvals), significance_level, ok)

    bo = -log10(bo)
    to = -log10(to)

    m = null_qvals[ok][cb_ok]

    ax.fill_between(m, bo, to, **band_kws)


def _subsample(pvalues, cutoff):
    from numpy import ones, percentile, log10, linspace, searchsorted, sum, where

    resolution = 500

    if len(pvalues) <= resolution:
        return ones(len(pvalues), dtype=bool)

    ok = pvalues <= percentile(pvalues, cutoff)
    nok = ~ok

    qv = -log10(pvalues[nok])
    qv_min = qv[-1]
    qv_max = qv[0]

    snok = sum(nok)

    resolution = min(snok, resolution)

    qv_chosen = linspace(qv_min, qv_max, resolution)
    pv_chosen = 10 ** (-qv_chosen)

    idx = searchsorted(pvalues[nok], pv_chosen)
    n = sum(nok)
    i = 0
    while i < len(idx) and idx[i] == n:
        i += 1
        idx = idx[i:]

    ok[where(nok)[0][idx]] = True

    ok[0] = True
    ok[-1] = True

    return ok
