import matplotlib.pyplot as plt

from nwlattice.base import NanowireLattice


def plot_index(wire: NanowireLattice, save_name: str = None, mirror: str = None,
               margins=(0.03, 2.0), show=False, save=True) -> None:
    index = wire.size.index
    indexer = wire.size.indexer

    lengths = []
    for i in range(len(index) - 1):
        lengths.append(index[i + 1] - index[i])

    styles = [
        dict(facecolor="coral"),
        dict(facecolor="indigo")
    ]

    fig, ax = plt.subplots(tight_layout=True, figsize=(14, 3))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    i_style = 0
    x0 = 0
    y0 = -0.5
    height = 2 * abs(y0)

    lengths_range = lengths
    if mirror:
        ax.plot([0.5, 0.5], [-height / 2, height], transform=ax.transAxes,
                color='k', linestyle=':')
        if mirror == "left":
            lengths_range = lengths[::-1] + lengths
        elif mirror == "right":
            lengths_range = lengths + lengths[::-1]
        else:
            raise ValueError("invalid `mirror` parameter {}".format(mirror))

    recs = []
    for length in lengths_range:
        i_style = (i_style + 1) % len(styles)
        recs.append(plt.Rectangle((x0, y0), length, height, **styles[i_style]))
        x0 += length

    for rec in recs:
        ax.add_patch(rec)

    ax.autoscale()
    ax.margins(*margins)

    if save_name is None:
        param_str = "_".join([str(_) for _ in indexer.params])
        save_name = "plot_index_%s_%s.png" % (wire.type_name, param_str)

    ax.text(margins[0], 0.64, r"m = %s | r = %s | $q_\mathrm{min}$ = %s | $q_\mathrm{max}$ = %s"
            % indexer.params, transform=ax.transAxes)

    ax.text(margins[0], 0.05, "scale = %f" % wire.size.scale, transform=ax.transAxes,
            fontsize=10)
    ax.text(1 - margins[0], 0.62, "~{:.2f}"
            .format(wire.size.length if mirror is None else 2 * wire.size.length),
            ha="right", transform=ax.transAxes)
    ax.text(1 - margins[0] + 0.005, 0.5, "{:.2f}".format(wire.size.width),
            va="center", rotation="vertical", transform=ax.transAxes)

    ax.text(margins[0], 0.3, lengths_range[:10], transform=ax.transAxes)
    ax.text(1 - margins[0], 0.3, lengths_range[-10:], ha="right", transform=ax.transAxes)

    if show:
        plt.show()
    if save:
        fig.savefig(save_name)
