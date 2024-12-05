import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.font_manager as font_manager
font_dir = ['paper/bitstream-charter-ttf/Charter/']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.font_manager.findSystemFonts(fontpaths='scipostphys-matplotlib', fontext='ttf')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'


plt.rcParams["figure.figsize"] = (9,9)
def plot_naive_unfold(pp, gen, rec, unfolded, name, bins=60,
               gen_weights=None, unfolded_weights=None, range=None, log=False, unit=None, density=False):

    y_t, bins = np.histogram(gen, bins=bins, range=range, weights=gen_weights)
    y_tr, _ = np.histogram(rec, bins=bins)
    y_g, _ = np.histogram(unfolded, bins=bins, weights = unfolded_weights)

    hists = [y_t, y_g, y_tr]
    hist_errors = [np.sqrt(y_t),  np.sqrt(y_g), np.sqrt(y_tr)]

    if density:
        integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
        scales = [1 / integral if integral != 0. else 1. for integral in integrals]
    else:
        scales = [1,1,1]

    FONTSIZE = 27
    labels = [r"$\text{gen}|_g$", r"$\text{unfolded}/\delta$", "rec"]
    colors = ["black","#A52A2A", "#0343DE"]
    dup_last = lambda a: np.append(a, a[-1])

    fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
    fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

    for y, y_err, scale, label, color in zip(hists, hist_errors, scales,
                                                 labels, colors):

        axs[0].step(bins, dup_last(y) * scale, label=label, color=color,
                        linewidth=1.0, where="post")
        axs[0].step(bins, dup_last(y + y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
        axs[0].step(bins, dup_last(y - y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
        axs[0].fill_between(bins, dup_last(y - y_err) * scale,
                                dup_last(y + y_err) * scale, facecolor=color,
                                alpha=0.3, step="post")

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.
        ratio_err[ratio_isnan] = 0.

        axs[1].step(bins, dup_last(ratio), linewidth=3.0, where="post", color=color)
        axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
        axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
        axs[1].fill_between(bins, dup_last(ratio - ratio_err),
                                dup_last(ratio + ratio_err), facecolor=color, alpha=0.25, step="post")

        delta = np.fabs(ratio - 1) * 100
        delta_err = ratio_err * 100

        markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                                  yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                                  linewidth=0, fmt=".", capsize=2, markersize=10)
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]


    for line in axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE-5).get_lines():
        line.set_linewidth(3.0)
    axs[0].set_ylabel("number of events", fontsize=FONTSIZE)

    if "p_{T" in name or log:
        axs[0].set_yscale("log")
        axs[0].set_ylim(1.e-5,5.e-2)



    # axs[1].set_ylabel(r"$\frac{\mathrm{unfolded}}{\mathrm{gen}}$",
    #                       fontsize=FONTSIZE)
    axs[1].set_ylabel(r"ratio",
                          fontsize=FONTSIZE)
    axs[1].set_yticks([0.95,1,1.05])
    axs[1].set_ylim([0.9, 1.1])
    axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
    axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

    if range:
        plt.xlim((range[0]+0.1,range[1]-0.1))
    #
    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[0].tick_params(axis="both", labelsize=FONTSIZE - 6)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE - 6)
    axs[2].tick_params(axis="both", labelsize=FONTSIZE - 6)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)


    plt.savefig(pp, format="pdf", bbox_inches='tight')
    plt.close()
def plot_reweighted_distribution(pp, true, fake, reweighted, name, bins=60,
                                 labels=None,true_weights=None, fake_weights=None, reweighted_weights=None, range=None, log=False, unit=None, density=False):

    if labels is None:
        labels = ["true", "fake", "reweighted"]
    y_t, bins = np.histogram(true, bins=bins, range=range, weights=true_weights)
    y_tr, _ = np.histogram(fake, bins=bins, weights= fake_weights)
    y_g, _ = np.histogram(reweighted, bins=bins, weights = reweighted_weights)

    hists = [y_t, y_g, y_tr]
    hist_errors = [np.sqrt(y_t),  np.sqrt(y_g), np.sqrt(y_tr)]
    dup_last = lambda a: np.append(a, a[-1])
    if density:
        integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
        scales = [1 / integral if integral != 0. else 1. for integral in integrals]
    else:
        scales = [1,1,1]

    FONTSIZE = 27
    if labels == None:
        labels = ["true", "fake", "reweighted"]

    colors = ["black", "#A52A2A", "#0343DE"]


    fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
    fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

    for y, y_err, scale, label, color in zip(hists, hist_errors, scales,
                                                 labels, colors):

        axs[0].step(bins, dup_last(y) * scale, label=label, color=color,
                        linewidth=1.0, where="post")
        axs[0].step(bins, dup_last(y + y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
        axs[0].step(bins, dup_last(y - y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
        axs[0].fill_between(bins, dup_last(y - y_err) * scale,
                                dup_last(y + y_err) * scale, facecolor=color,
                                alpha=0.3, step="post")

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.
        ratio_err[ratio_isnan] = 0.

        axs[1].step(bins, dup_last(ratio), linewidth=3.0, where="post", color=color)
        axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
        axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
        axs[1].fill_between(bins, dup_last(ratio - ratio_err),
                                dup_last(ratio + ratio_err), facecolor=color, alpha=0.25, step="post")

        delta = np.fabs(ratio - 1) * 100
        delta_err = ratio_err * 100

        markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                                  yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                                  linewidth=0, fmt=".", capsize=2, markersize=10)
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]


    for line in axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE-5).get_lines():
        line.set_linewidth(3.0)
    axs[0].set_ylabel("number of events", fontsize=FONTSIZE)

    if "p_{T" in name or log:
        axs[0].set_yscale("log")
        axs[0].set_ylim(1.e-5,5.e-2)



    # axs[1].set_ylabel(r"$\frac{\mathrm{unfolded}}{\mathrm{gen}}$",
    #                       fontsize=FONTSIZE)
    axs[1].set_ylabel(r"ratio",
                          fontsize=FONTSIZE)
    axs[1].set_yticks([0.95,1,1.05])
    axs[1].set_ylim([0.9, 1.1])
    axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
    axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

    if range:
        plt.xlim((range[0]+0.1,range[1]-0.1))
    #
    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[0].tick_params(axis="both", labelsize=FONTSIZE - 6)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE - 6)
    axs[2].tick_params(axis="both", labelsize=FONTSIZE - 6)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)


    plt.savefig(pp, format="pdf", bbox_inches='tight')
    plt.close()


def plot_prior_unfold(pp, gen, prior, unfolded, name, bins=60,
               gen_weights=None,prior_weights=None, unfolded_weights=None, range=None, log=False, unit=None, density=False):

    y_t, bins = np.histogram(gen, bins=bins, range=range, weights=gen_weights)
    y_tr, _ = np.histogram(prior, bins=bins, weights= prior_weights)
    y_g, _ = np.histogram(unfolded, bins=bins, weights = unfolded_weights)

    hists = [y_t, y_g, y_tr]
    hist_errors = [np.sqrt(y_t),  np.sqrt(y_g), np.sqrt(y_tr)]

    if density:
        integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
        scales = [1 / integral if integral != 0. else 1. for integral in integrals]
    else:
        scales = [1,1,1]

    FONTSIZE = 27
    labels = [r"$\text{gen}|_{g,r}$", "unfolded", "prior"]
    colors = ["black","#A52A2A", "#0343DE"]
    dup_last = lambda a: np.append(a, a[-1])

    fig1, axs = plt.subplots(3, 1, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.00})
    fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

    for y, y_err, scale, label, color in zip(hists, hist_errors, scales,
                                                 labels, colors):

        axs[0].step(bins, dup_last(y) * scale, label=label, color=color,
                        linewidth=1.0, where="post")
        axs[0].step(bins, dup_last(y + y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
        axs[0].step(bins, dup_last(y - y_err) * scale, color=color,
                        alpha=0.5, linewidth=0.5, where="post")
        axs[0].fill_between(bins, dup_last(y - y_err) * scale,
                                dup_last(y + y_err) * scale, facecolor=color,
                                alpha=0.3, step="post")

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.
        ratio_err[ratio_isnan] = 0.

        axs[1].step(bins, dup_last(ratio), linewidth=3.0, where="post", color=color)
        axs[1].step(bins, dup_last(ratio + ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
        axs[1].step(bins, dup_last(ratio - ratio_err), color=color, alpha=0.5,
                        linewidth=0.5, where="post")
        axs[1].fill_between(bins, dup_last(ratio - ratio_err),
                                dup_last(ratio + ratio_err), facecolor=color, alpha=0.25, step="post")

        delta = np.fabs(ratio - 1) * 100
        delta_err = ratio_err * 100

        markers, caps, bars = axs[2].errorbar((bins[:-1] + bins[1:]) / 2, delta,
                                                  yerr=delta_err, ecolor=color, color=color, elinewidth=0.5,
                                                  linewidth=0, fmt=".", capsize=2, markersize=10)
        [cap.set_alpha(0.5) for cap in caps]
        [bar.set_alpha(0.5) for bar in bars]


    for line in axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE-5).get_lines():
        line.set_linewidth(3.0)
    axs[0].set_ylabel("number of events", fontsize=FONTSIZE)

    if "p_{T" in name or log:
        axs[0].set_yscale("log")
        axs[0].set_ylim(1.e-5,5.e-2)



    # axs[1].set_ylabel(r"$\frac{\mathrm{unfolded}}{\mathrm{gen}}$",
    #                       fontsize=FONTSIZE)
    axs[1].set_ylabel(r"ratio",
                          fontsize=FONTSIZE)
    axs[1].set_yticks([0.95,1,1.05])
    axs[1].set_ylim([0.9, 1.1])
    axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
    axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
    axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

    if range:
        plt.xlim((range[0]+0.1,range[1]-0.1))
    #
    axs[2].set_ylim((0.05, 20))
    axs[2].set_yscale("log")
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    axs[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    axs[2].set_ylabel(r"$\delta [\%]$", fontsize=FONTSIZE)

    axs[0].tick_params(axis="both", labelsize=FONTSIZE - 6)
    axs[1].tick_params(axis="both", labelsize=FONTSIZE - 6)
    axs[2].tick_params(axis="both", labelsize=FONTSIZE - 6)
    plt.xlabel(r"${%s}$ %s" % (name, ("" if unit is None else f"[{unit}]")),
                   fontsize=FONTSIZE)


    plt.savefig(pp, format="pdf", bbox_inches='tight')
    plt.close()