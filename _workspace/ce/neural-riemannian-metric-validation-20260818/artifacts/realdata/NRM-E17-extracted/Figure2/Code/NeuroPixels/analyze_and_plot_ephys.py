
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import mannwhitneyu, wilcoxon
from pathlib import Path
from matplotlib.transforms import Bbox

# configuration
FIGURE_HEIGHT = 3
FONT_SIZE = 15
FONT_FAMILY = "Arial"
FIGURE_DPI = 300
FIGURE_SIZE = (3, 3)

COLOR_FR = "k"
COLOR_BR = "#BB5566"
COLOR_EFF = "#004488"
COLORS_MAIN = [COLOR_FR, COLOR_BR, COLOR_EFF]

CONFIG = {
    # path relative to this script
    "output_dir": Path(__file__).parent / "PLOTS_STATISTICS",
    "effect_size": True,
    "min_backprop_distance": 50,

    "font_size": FONT_SIZE,
    "figure_dpi": FIGURE_DPI,
    "modulation_threshold_percent": 5.0,
    "color_palette": COLORS_MAIN,
}

# helper functions

def my_ceil(a, precision=0):
    """rounds up a number to a specified precision.
    
    args:
        a: input number or array.
        precision: number of decimal places."""
    return np.round(a + 0.5 * 10 ** (-precision), precision)

def dropNaNs(arr1, arr2):
    """removes indices where either array has nan. We don't want to include neurons that drifted away.
    
    args:
        arr1: first array.
        arr2: second array."""
    good_idxs = np.where(~np.isnan(arr1) & ~np.isnan(arr2))
    return arr1[good_idxs], arr2[good_idxs]

def getAsterisks(p):
    """returns significance asterisks for a given p-value.
    
    args:
        p: p-value from statistical test."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def full_extent(ax, pad=0.0):
    """calculates the full extent of an axes object including labels.
    
    args:
        ax: matplotlib axes object.
        pad: padding to add."""
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def formatScatterPlot(ax, ylim, axeslabel, log=False, is_effect_size=False):
    """formats a scatter plot with specific styles.
    
    args:
        ax: matplotlib axes object.
        ylim: limits for both axes.
        axeslabel: tuple of (xlabel, ylabel).
        log: whether to use log scale.
        is_effect_size: boolean to apply effect size specific formatting."""
    ax.spines[["top", "right"]].set_visible(False)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if is_effect_size:
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_yticklabels([-1, 1])
        ax.set_xticklabels([-1, 1])
    else:
        ax.set_ylim(ylim)
        ax.set_xlim(ylim)
        ax.set_xticks(ylim)
        ax.set_yticks(ylim)
        ax.set_yticklabels([0, ylim[1]])
        ax.set_xticklabels([0, ylim[1]])
    ax.set_xlabel(axeslabel[0])
    ax.set_ylabel(axeslabel[1])

def formatBoxPlot(ax, B, xlabels, is_effect_size=False):
    """formats a box plot and returns y-axis limits.
    
    args:
        ax: matplotlib axes object.
        b: boxplot dictionary.
        xlabels: list of x-axis labels.
        is_effect_size: boolean for effect size formatting."""
    if is_effect_size:
        this_ylim = [-1, 1]
    else:
        whiskertop = np.nanmax([item.get_ydata()[1] for item in B["whiskers"]])
        this_ylim = [0, my_ceil(whiskertop, 1)]
    ax.set_xticklabels(xlabels, rotation=60)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_ylim(this_ylim)
    ax.set_yticks([], [])
    return this_ylim

# plotting functions

def plotData(
    frs_ctrl,
    frs_DCZ,
    brs_ctrl,
    brs_DCZ,
    header,
    outputPath,
    colors=["k", "#BB5566", "#004488"],
    saveSVG=False,
    orientation="vertical",
    effect_size=True,
    savePDF=True,
    saveIndividualPlots=False,
):
    """plots firing rate and burst rate comparisons (boxplots and scatter).
    
    args:
        frs_ctrl: firing rates control.
        frs_dcz: firing rates dcz.
        brs_ctrl: burst rates control.
        brs_dcz: burst rates dcz.
        header: title/filename prefix.
        outputpath: directory to save plots.
        colors: list of colors for fr, br, and effect size.
        savesvg: whether to save as svg.
        orientation: plot orientation.
        effect_size: whether to plot effect sizes.
        savepdf: whether to save as pdf.
        saveindividualplots: whether to save separate plots."""
    frs_ctrl_clean = frs_ctrl.copy()
    frs_DCZ_clean = frs_DCZ.copy()
    brs_ctrl_clean = brs_ctrl.copy()
    brs_DCZ_clean = brs_DCZ.copy()

    frs_ctrl_clean[frs_ctrl_clean == 0] = np.nan
    frs_DCZ_clean[frs_DCZ_clean == 0] = np.nan
    brs_ctrl_clean[brs_ctrl_clean == 0] = np.nan
    brs_DCZ_clean[brs_DCZ_clean == 0] = np.nan

    avgBFCtrl = brs_ctrl_clean / frs_ctrl_clean
    avgBFDCZ = brs_DCZ_clean / frs_DCZ_clean

    def getEffectSize(A, B):
        return (A - B) / (A + B)

    effFR = getEffectSize(frs_ctrl_clean, frs_DCZ_clean)
    effBR = getEffectSize(brs_ctrl_clean, brs_DCZ_clean)

    w, p_FR = wilcoxon(frs_ctrl_clean, frs_DCZ_clean, nan_policy="omit")
    w, p_BR = wilcoxon(brs_ctrl_clean, brs_DCZ_clean, nan_policy="omit")
    if effect_size:
        w, p_BF = wilcoxon(effFR, effBR, nan_policy="omit")
    else:
        w, p_BF = wilcoxon(avgBFCtrl, avgBFDCZ, nan_policy="omit")

    frs_ctrl_plot, frs_DCZ_plot = dropNaNs(frs_ctrl_clean, frs_DCZ_clean)
    brs_ctrl_plot, brs_DCZ_plot = dropNaNs(brs_ctrl_clean, brs_DCZ_clean)
    if effect_size:
        effFR_plot, effBR_plot = dropNaNs(effFR, effBR)
    else:
        avgBFCtrl_plot, avgBFDCZ_plot = dropNaNs(avgBFCtrl, avgBFDCZ)

    color_FR, color_BR, color_BF = colors

    fig, ax = plt.subplots(
        ncols=2,
        nrows=3,
        figsize=(4, FIGURE_HEIGHT * 2.5),
        width_ratios=[3, 2],
        constrained_layout=True,
    )
    ax = ax.flatten()
    axesCoords = range(6)


    B = ax[axesCoords[1]].boxplot(
        [frs_ctrl_plot, frs_DCZ_plot], widths=0.7, sym="", patch_artist=True,
        boxprops=dict(facecolor=color_FR, color=color_FR, alpha=0.2),
        whiskerprops=dict(color=color_FR), capprops=dict(color=color_FR), medianprops=dict(color=color_FR),
    )
    ax[axesCoords[1]].text(
        0.5, 0.992, f"{getAsterisks(p_FR)}", ha="center", va="center",
        transform=ax[axesCoords[1]].transAxes, fontsize=10, weight="bold"
    )
    this_ylim = formatBoxPlot(ax[axesCoords[1]], B, ["Ctrl", "DCZ"])

    jitter_ctrl = np.random.normal(0, 0.1, len(frs_ctrl_plot))
    jitter_dcz = np.random.normal(0, 0.1, len(frs_DCZ_plot))
    ax[axesCoords[1]].scatter(1 + jitter_ctrl, frs_ctrl_plot, c=color_FR, s=15, alpha=0.3, zorder=10)
    ax[axesCoords[1]].scatter(2 + jitter_dcz, frs_DCZ_plot, c=color_FR, s=15, alpha=0.3, zorder=10)
    for i in range(len(frs_ctrl_plot)):
        ax[axesCoords[1]].plot(
            [1.15, 1.85], [frs_ctrl_plot[i], frs_DCZ_plot[i]], 
            color=color_FR, alpha=0.2, linewidth=0.5, zorder=5
        )

    ax[axesCoords[0]].scatter(frs_ctrl_plot, frs_DCZ_plot, c=color_FR, s=10, alpha=0.4)
    ax[axesCoords[0]].plot(this_ylim, this_ylim, c=color_FR, linestyle="--", alpha=0.2)
    formatScatterPlot(ax[axesCoords[0]], this_ylim, ["Ctrl FR (Hz)", "DCZ FR (Hz)"])


    B = ax[axesCoords[3]].boxplot(
        [brs_ctrl_plot, brs_DCZ_plot], widths=0.7, sym="", patch_artist=True,
        boxprops=dict(facecolor=color_BR, color=color_BR, alpha=0.2),
        whiskerprops=dict(color=color_BR), capprops=dict(color=color_BR), medianprops=dict(color=color_BR),
    )
    ax[axesCoords[3]].text(
        0.5, 0.992, f"{getAsterisks(p_BR)}", ha="center", va="center",
        transform=ax[axesCoords[3]].transAxes, fontsize=10, weight="bold"
    )
    this_ylim = formatBoxPlot(ax[axesCoords[3]], B, ["Ctrl", "DCZ"])

    jitter_ctrl = np.random.normal(0, 0.1, len(brs_ctrl_plot))
    jitter_dcz = np.random.normal(0, 0.1, len(brs_DCZ_plot))
    ax[axesCoords[3]].scatter(1 + jitter_ctrl, brs_ctrl_plot, c=color_BR, s=15, alpha=0.3, zorder=10)
    ax[axesCoords[3]].scatter(2 + jitter_dcz, brs_DCZ_plot, c=color_BR, s=15, alpha=0.3, zorder=10)
    for i in range(len(brs_ctrl_plot)):
        ax[axesCoords[3]].plot(
            [1.15, 1.85], [brs_ctrl_plot[i], brs_DCZ_plot[i]], 
            color=color_BR, alpha=0.2, linewidth=0.5, zorder=5
        )

    ax[axesCoords[2]].scatter(brs_ctrl_plot, brs_DCZ_plot, c=color_BR, s=10, alpha=0.4)
    ax[axesCoords[2]].plot(this_ylim, this_ylim, c=color_BR, linestyle="--", alpha=0.2)
    formatScatterPlot(ax[axesCoords[2]], this_ylim, ["Ctrl BR (Hz)", "DCZ BR (Hz)"])


    if effect_size:
        d1, d2 = effFR_plot, effBR_plot
    else:
        d1, d2 = avgBFCtrl_plot, avgBFDCZ_plot

    B = ax[axesCoords[5]].boxplot(
        [d1, d2], widths=0.7, sym="", patch_artist=True,
        boxprops=dict(facecolor=color_BF, color=color_BF, alpha=0.2),
        whiskerprops=dict(color=color_BF), capprops=dict(color=color_BF), medianprops=dict(color=color_BF),
    )
    ax[axesCoords[5]].text(
        0.5, 0.992, f"{getAsterisks(p_BF)}", ha="center", va="center",
        transform=ax[axesCoords[5]].transAxes, fontsize=10, weight="bold"
    )
    x_ticklabels = ["FR", "BR"] if effect_size else ["Ctrl", "DCZ"]
    this_ylim = formatBoxPlot(ax[axesCoords[5]], B, x_ticklabels, is_effect_size=effect_size)

    jitter_d1 = np.random.normal(0, 0.1, len(d1))
    jitter_d2 = np.random.normal(0, 0.1, len(d2))
    ax[axesCoords[5]].scatter(1 + jitter_d1, d1, c=color_BF, s=15, alpha=0.3, zorder=10)
    ax[axesCoords[5]].scatter(2 + jitter_d2, d2, c=color_BF, s=15, alpha=0.3, zorder=10)
    for i in range(len(d1)):
        ax[axesCoords[5]].plot(
            [1.15, 1.85], [d1[i], d2[i]], 
            color=color_BF, alpha=0.2, linewidth=0.5, zorder=5
        )

    if effect_size:
        ax[axesCoords[4]].scatter(effFR_plot, effBR_plot, c=color_BF, s=10, alpha=0.4)
        ax[axesCoords[4]].plot([-1, 1], [-1, 1], c=color_BF, linestyle="--", alpha=0.2)
    else:
        ax[axesCoords[4]].scatter(avgBFCtrl_plot, avgBFDCZ_plot, c=color_BF, s=10, alpha=0.4)
        ax[axesCoords[4]].plot(this_ylim, this_ylim, c=color_BF, linestyle="--", alpha=0.2)
    
    yStr = "Effect size BR" if effect_size else "Ctrl BF"
    xStr = "Effect size FR" if effect_size else "DCZ BF"
    ax[axesCoords[4]].set_xlabel(f"{xStr}")
    ax[axesCoords[4]].set_ylabel(f"{yStr}")
    formatScatterPlot(ax[axesCoords[4]], this_ylim, [xStr, yStr], log=False, is_effect_size=effect_size)

    if savePDF:
        plt.savefig(outputPath / f"{header}.pdf", transparent=True)
    plt.close()
    
    return {
        "n_neurons": len(frs_ctrl_clean),
        "fr_ctrl_mean": np.nanmean(frs_ctrl_clean), "fr_ctrl_std": np.nanstd(frs_ctrl_clean),
        "fr_dcz_mean": np.nanmean(frs_DCZ_clean), "fr_dcz_std": np.nanstd(frs_DCZ_clean),
        "fr_change_pct": ((np.nanmean(frs_DCZ_clean) - np.nanmean(frs_ctrl_clean)) / np.nanmean(frs_ctrl_clean)) * 100,
        "p_FR": p_FR,
        "br_ctrl_mean": np.nanmean(brs_ctrl_clean), "br_ctrl_std": np.nanstd(brs_ctrl_clean),
        "br_dcz_mean": np.nanmean(brs_DCZ_clean), "br_dcz_std": np.nanstd(brs_DCZ_clean),
        "br_change_pct": ((np.nanmean(brs_DCZ_clean) - np.nanmean(brs_ctrl_clean)) / np.nanmean(brs_ctrl_clean)) * 100,
        "p_BR": p_BR,
        "p_BR": p_BR,
        "fr_eff_mean": np.nanmean(effFR), "fr_eff_std": np.nanstd(effFR),
        "br_eff_mean": np.nanmean(effBR), "br_eff_std": np.nanstd(effBR),
        "p_eff": p_BF
    }

def plot_interneurons_combined(cluster_info, interneuron_cluster_info, output_path):
    """plots interneuron firing rate changes and compares effect sizes.
    
    args:
        cluster_info: dataframe for pyramidal neurons (for comparison).
        interneuron_cluster_info: dataframe for interneurons.
        output_path: directory to save plots."""
    interneurons = interneuron_cluster_info
    other_neurons = cluster_info

    if len(interneurons) == 0:
        return {}

    def getEffectSize(A, B):
        return (A - B) / (A + B)

    interneuron_frs_ctrl = interneurons["avgFRsCtrl"].values.copy()
    interneuron_frs_dcz = interneurons["avgFRsDCZ"].values.copy()
    interneuron_frs_ctrl[interneuron_frs_ctrl == 0] = np.nan
    interneuron_frs_dcz[interneuron_frs_dcz == 0] = np.nan
    interneuron_eff = getEffectSize(interneuron_frs_ctrl, interneuron_frs_dcz)

    other_frs_ctrl = other_neurons["avgFRsCtrl"].values.copy()
    other_frs_dcz = other_neurons["avgFRsDCZ"].values.copy()
    other_frs_ctrl[other_frs_ctrl == 0] = np.nan
    other_frs_dcz[other_frs_dcz == 0] = np.nan
    other_eff = getEffectSize(other_frs_ctrl, other_frs_dcz)

    interneuron_eff = interneuron_eff[~np.isnan(interneuron_eff)]
    other_eff = other_eff[~np.isnan(other_eff)]

    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(4, FIGURE_HEIGHT * 2),
        width_ratios=[3, 2],
        constrained_layout=True,
    )
    ax = ax.flatten()


    frs_ctrl_inter = interneurons["avgFRsCtrl"].values
    frs_dcz_inter = interneurons["avgFRsDCZ"].values
    w, p_FR = wilcoxon(frs_ctrl_inter, frs_dcz_inter, nan_policy="omit")
    
    frs_ctrl_inter_plot, frs_dcz_inter_plot = dropNaNs(frs_ctrl_inter, frs_dcz_inter)

    ax[0].scatter(frs_ctrl_inter_plot, frs_dcz_inter_plot, c="k", s=10, alpha=0.4)
    ylim = [0, max(np.nanmax(frs_ctrl_inter_plot), np.nanmax(frs_dcz_inter_plot)) * 1.1]
    ax[0].plot(ylim, ylim, c="k", linestyle="--", alpha=0.2)
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(ylim)
    ax[0].set_xticks(ylim)
    ax[0].set_yticks(ylim)
    ax[0].set_yticklabels([0, int(ylim[1])])
    ax[0].set_xticklabels([0, int(ylim[1])])
    ax[0].set_xlabel("Ctrl FR (Hz)")
    ax[0].set_ylabel("DCZ FR (Hz)")

    B = ax[1].boxplot(
        [frs_ctrl_inter_plot, frs_dcz_inter_plot], widths=0.7, sym="", patch_artist=True,
        boxprops=dict(facecolor="k", color="k", alpha=0.2),
        whiskerprops=dict(color="k"), capprops=dict(color="k"), medianprops=dict(color="k"),
    )
    ax[1].text(
        0.5, 0.992, f"{getAsterisks(p_FR)}", ha="center", va="center",
        transform=ax[1].transAxes, fontsize=10, weight="bold"
    )
    ax[1].set_xticklabels(["Ctrl", "DCZ"], rotation=60)
    ax[1].spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax[1].set_ylim(ylim)
    ax[1].set_yticks([], [])

    np.random.seed(42)
    jitter_ctrl = np.random.normal(0, 0.1, len(frs_ctrl_inter_plot))
    jitter_dcz = np.random.normal(0, 0.1, len(frs_dcz_inter_plot))
    ax[1].scatter(1 + jitter_ctrl, frs_ctrl_inter_plot, c="k", s=15, alpha=0.3, zorder=10)
    ax[1].scatter(2 + jitter_dcz, frs_dcz_inter_plot, c="k", s=15, alpha=0.3, zorder=10)
    for i in range(len(frs_ctrl_inter_plot)):
        ax[1].plot(
            [1.15, 1.85], [frs_ctrl_inter_plot[i], frs_dcz_inter_plot[i]],
            color="k", alpha=0.2, linewidth=0.5, zorder=5
        )


    w, p_eff = mannwhitneyu(interneuron_eff, other_eff, nan_policy="omit")

    ax[2].text(
        0.5, 0.5, "Not applicable\n(unpaired groups)", ha="center", va="center",
        transform=ax[2].transAxes, fontsize=12, color="gray"
    )
    ax[2].axis("off")

    COLOR_INT_EFF = "#004488"
    B = ax[3].boxplot(
        [interneuron_eff, other_eff], widths=0.7, sym="", patch_artist=True,
        boxprops=dict(facecolor=COLOR_INT_EFF, color=COLOR_INT_EFF, alpha=0.2),
        whiskerprops=dict(color=COLOR_INT_EFF), capprops=dict(color=COLOR_INT_EFF), medianprops=dict(color=COLOR_INT_EFF),
    )
    ax[3].text(
        0.5, 0.992, f"{getAsterisks(p_eff)}", ha="center", va="center",
        transform=ax[3].transAxes, fontsize=10, weight="bold"
    )
    ax[3].set_xticklabels(["Inter", "Other"], rotation=60)
    this_ylim = [-1, 1]
    ax[3].spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax[3].set_ylim(this_ylim)
    ax[3].set_yticks([], [])

    np.random.seed(42)
    jitter_int = np.random.normal(0, 0.05, len(interneuron_eff))
    jitter_other = np.random.normal(0, 0.05, len(other_eff))
    ax[3].scatter(1 + jitter_int, interneuron_eff, c=COLOR_INT_EFF, s=15, alpha=0.3, zorder=10)
    ax[3].scatter(2 + jitter_other, other_eff, c=COLOR_INT_EFF, s=15, alpha=0.3, zorder=10)

    plt.savefig(output_path / "L5_interneuron_fr.pdf", transparent=True)
    plt.close()
    
    return {
        "n_neurons": len(interneurons),
        "fr_ctrl_mean": np.nanmean(frs_ctrl_inter), "fr_ctrl_std": np.nanstd(frs_ctrl_inter),
        "fr_dcz_mean": np.nanmean(frs_dcz_inter), "fr_dcz_std": np.nanstd(frs_dcz_inter),
        "fr_change_pct": ((np.nanmean(frs_dcz_inter) - np.nanmean(frs_ctrl_inter)) / np.nanmean(frs_ctrl_inter)) * 100,
        "p_FR": p_FR,
        "p_FR": p_FR,
        "fr_eff_mean": np.nanmean(interneuron_eff), "fr_eff_std": np.nanstd(interneuron_eff),
        "p_eff": p_eff
    }

def plot_backpropagation_features(cluster_info, output_path):
    """plots backpropagation distance and speed changes.
    
    args:
        cluster_info: dataframe containing neuronal metrics.
        output_path: directory to save plots."""
    valid_mask = cluster_info[[
        "window_0_propagation_distance_um", "window_1_propagation_distance_um",
        "window_0_speed_above", "window_1_speed_above"
    ]].notna().all(axis=1)
    valid_data = cluster_info[valid_mask]

    min_distance = CONFIG["min_backprop_distance"]
    distance_mask = (valid_data["window_0_propagation_distance_um"] >= min_distance) & \
                    (valid_data["window_1_propagation_distance_um"] >= min_distance)
    valid_data = valid_data[distance_mask]
    
    if len(valid_data) == 0:
        return []

    features = [
        ("window_0_propagation_distance_um", "window_1_propagation_distance_um", "backprop\nDistance", "backprop_distance"),
        ("window_0_speed_above", "window_1_speed_above", "backprop\nSpeed", "backprop_speed")
    ]

    stats_list = []
    
    COLOR_BP = "#BB5566"

    for ctrl_col, dcz_col, feature_name, filename in features:
        fig, ax = plt.subplots(
            ncols=2,
            nrows=1,
            figsize=(4, FIGURE_HEIGHT * 2.5 / 3),
            width_ratios=[3, 2],
            constrained_layout=True,
        )
        
        ctrl_values = valid_data[ctrl_col].values
        dcz_values = valid_data[dcz_col].values

        stat, p_value = wilcoxon(ctrl_values, dcz_values, nan_policy="omit")

        ax[0].scatter(ctrl_values, dcz_values, c=COLOR_BP, s=10, alpha=0.4)
        
        is_distance = "distance" in feature_name.lower()
        lower_lim = min_distance if is_distance else 0
        ylim = [lower_lim, max(np.nanmax(ctrl_values), np.nanmax(dcz_values)) * 1.1]
        
        ax[0].plot(ylim, ylim, c=COLOR_BP, linestyle="--", alpha=0.2)
        
        ax[0].spines[["top", "right"]].set_visible(False)
        ax[0].set_ylim(ylim)
        ax[0].set_xlim(ylim)
        ax[0].set_xticks(ylim)
        ax[0].set_yticks(ylim)
        ax[0].set_yticklabels([int(ylim[0]), int(ylim[1])])
        ax[0].set_xticklabels([int(ylim[0]), int(ylim[1])])
        ax[0].set_xlabel(f"Ctrl {feature_name.split()[-1]}")
        ax[0].set_ylabel(f"DCZ {feature_name.split()[-1]}")

        B = ax[1].boxplot(
            [ctrl_values, dcz_values], widths=0.7, sym="", patch_artist=True,
            boxprops=dict(facecolor=COLOR_BP, color=COLOR_BP, alpha=0.2),
            whiskerprops=dict(color=COLOR_BP), capprops=dict(color=COLOR_BP), medianprops=dict(color=COLOR_BP),
        )
        ax[1].text(
            0.5, 0.992, f"{getAsterisks(p_value)}", ha="center", va="center",
            transform=ax[1].transAxes, fontsize=15, weight="bold"
        )
        
        whiskertop = np.nanmax([item.get_ydata()[1] for item in B["whiskers"]])
        if "distance" in feature_name.lower():
            bottom_limit = max(0, min_distance)
            this_ylim = [bottom_limit, my_ceil(whiskertop, 1)]
        else:
            this_ylim = [0, my_ceil(whiskertop, 1)]
            
        ax[1].set_xticklabels(["Ctrl", "DCZ"], rotation=60)
        ax[1].spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax[1].set_ylim(this_ylim)
        ax[1].set_yticks([], [])

        np.random.seed(42)
        jitter_ctrl = np.random.normal(0, 0.1, len(ctrl_values))
        jitter_dcz = np.random.normal(0, 0.1, len(dcz_values))
        ax[1].scatter(1 + jitter_ctrl, ctrl_values, c=COLOR_BP, s=15, alpha=0.3, zorder=10)
        ax[1].scatter(2 + jitter_dcz, dcz_values, c=COLOR_BP, s=15, alpha=0.3, zorder=10)
        for i in range(len(ctrl_values)):
            ax[1].plot(
                [1.15, 1.85], [ctrl_values[i], dcz_values[i]],
                color=COLOR_BP, alpha=0.2, linewidth=0.5, zorder=5
            )

        plt.savefig(output_path / f"{filename}.pdf", transparent=True)
        plt.close()

        ctrl_mean, ctrl_std = np.nanmean(ctrl_values), np.nanstd(ctrl_values)
        dcz_mean, dcz_std = np.nanmean(dcz_values), np.nanstd(dcz_values)
        change_pct = ((dcz_mean - ctrl_mean) / ctrl_mean) * 100 if ctrl_mean != 0 else 0
        stats_list.append({
            "feature": feature_name.replace("\n", " "),
            "ctrl_mean": ctrl_mean, "ctrl_std": ctrl_std,
            "dcz_mean": dcz_mean, "dcz_std": dcz_std,
            "change_pct": change_pct,
            "p_value": p_value
        })
    return stats_list

def create_modulation_pie_chart(data, ctrl_col, dcz_col, rate_type, filename, threshold, output_path):
    """creates a pie chart showing proportion of modulated neurons.
    
    args:
        data: dataframe with neuronal data.
        ctrl_col: column name for control values.
        dcz_col: column name for dcz values.
        rate_type: label for the rate type being plotted.
        filename: output filename.
        threshold: percentage threshold for modulation.
        output_path: directory to save plots."""
    # drop nans for accurate pie chart
    data = data.dropna(subset=[ctrl_col, dcz_col])
    data = data[data[ctrl_col] != 0] # also exclude true zeros as we don't want to include neurons that drifted away

    percentage_change = (data[dcz_col] - data[ctrl_col]) / data[ctrl_col] * 100
    positive_mod = (percentage_change > threshold).sum()
    negative_mod = (percentage_change < -threshold).sum()
    not_modulated = len(data) - positive_mod - negative_mod
    
    sizes = [positive_mod, negative_mod, not_modulated]
    labels = ["Positively\nmodulated", "Negatively\nmodulated", "Not\nmodulated"]
    colors = ["#cc6677", "#88ccee", "#C0C0C0"]

    if sum(sizes) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda pct: f"{int(pct / 100 * sum(sizes))}",
        startangle=90,
        textprops={"color": "black", "fontsize": CONFIG["font_size"]}
    )

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")

    ax.set_title(f"{rate_type} Modulation\n(Threshold: ±{threshold}%)\nn={sum(sizes)}", 
                 fontsize=CONFIG["font_size"], fontweight="bold", pad=20)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_path / f"{filename}.pdf", bbox_inches="tight", transparent=True)
    plt.close()

# main execution

def main():
    """main execution function handling data loading, processing, and plotting."""
    # path relative to this script
    extracted_data_path = Path(__file__).parent / "extracted_data" / "extracted_neural_data.pkl"
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_pickle(extracted_data_path)

    column_mapping = {
        "neuron_id": "neuron_uid",
        # global rates
        "tonic_fr_ctrl": "global_avgFRsCtrl",
        "tonic_fr_dcz": "global_avgFRsDCZ",
        "burst_rate_ctrl": "global_avgBRsCtrl",
        "burst_rate_dcz": "global_avgBRsDCZ",
        # puff rates
        "tonic_fr_ctrl_puff": "avgFRsCtrl",
        "tonic_fr_dcz_puff": "avgFRsDCZ",
        "burst_rate_ctrl_puff": "avgBRsCtrl",
        "burst_rate_dcz_puff": "avgBRsDCZ",
        # backprop
        "backprop_dist_ctrl_puff": "window_0_propagation_distance_um",
        "backprop_dist_dcz_puff": "window_1_propagation_distance_um",
        "backprop_speed_ctrl_puff": "window_0_speed_above",
        "backprop_speed_dcz_puff": "window_1_speed_above"
    }
    df = df.rename(columns=column_mapping)
    
    df_pyr = df[df["is_interneuron"] == False].copy()
    df_int = df[df["is_interneuron"] == True].copy()
    

    print("Generating Pyramidal Combined Plot (Main)...")
    pyr_stats = plotData(
        df_pyr["avgFRsCtrl"].values,
        df_pyr["avgFRsDCZ"].values,
        df_pyr["avgBRsCtrl"].values,
        df_pyr["avgBRsDCZ"].values,
        "L5_pyramidals_puff_ctrl_DCZ",
        CONFIG["output_dir"],
        saveIndividualPlots=True
    )


    print("Generating Interneuron Plot...")
    int_stats = plot_interneurons_combined(df_pyr, df_int, CONFIG["output_dir"])


    print("Generating Backpropagation Plots...")
    bp_stats_list = plot_backpropagation_features(df_pyr, CONFIG["output_dir"])


    print("Generating Pie Charts...")
    df_pyr_L5 = df_pyr[df_pyr["layer"] == "5"]
    print(f"L5 Pyramidal neurons for Pie Charts: {len(df_pyr_L5)}")
    
    # puff modulation (local window)
    create_modulation_pie_chart(df_pyr_L5, "avgFRsCtrl", "avgFRsDCZ", "Firing Rate (Puff Window)", "modulation_piechart_pyr_fr_puff", CONFIG["modulation_threshold_percent"], CONFIG["output_dir"])
    create_modulation_pie_chart(df_pyr_L5, "avgBRsCtrl", "avgBRsDCZ", "Burst Rate (Puff Window)", "modulation_piechart_pyr_br_puff", CONFIG["modulation_threshold_percent"], CONFIG["output_dir"])

    # global modulation (whole session)
    create_modulation_pie_chart(df_pyr_L5, "global_avgFRsCtrl", "global_avgFRsDCZ", "Firing Rate (Global)", "modulation_piechart_pyr_fr_global", CONFIG["modulation_threshold_percent"], CONFIG["output_dir"])
    create_modulation_pie_chart(df_pyr_L5, "global_avgBRsCtrl", "global_avgBRsDCZ", "Burst Rate (Global)", "modulation_piechart_pyr_br_global", CONFIG["modulation_threshold_percent"], CONFIG["output_dir"])


    # save statistics text file
    stats_file = CONFIG["output_dir"] / "statistics.txt"
    with open(stats_file, "w") as f:
        f.write("=== L5 PYRAMIDAL NEURONS (PUFF) ===\n")
        f.write(f"n = {pyr_stats['n_neurons']}\n")
        f.write(f"FR: Ctrl = {pyr_stats['fr_ctrl_mean']:.2f} ± {pyr_stats['fr_ctrl_std']:.2f}, DCZ = {pyr_stats['fr_dcz_mean']:.2f} ± {pyr_stats['fr_dcz_std']:.2f}, Change = {pyr_stats['fr_change_pct']:.2f}%, p = {pyr_stats['p_FR']}\n")
        f.write(f"BR: Ctrl = {pyr_stats['br_ctrl_mean']:.2f} ± {pyr_stats['br_ctrl_std']:.2f}, DCZ = {pyr_stats['br_dcz_mean']:.2f} ± {pyr_stats['br_dcz_std']:.2f}, Change = {pyr_stats['br_change_pct']:.2f}%, p = {pyr_stats['p_BR']}\n")
        f.write(f"Effect Size: FR = {pyr_stats['fr_eff_mean']:.2f} ± {pyr_stats['fr_eff_std']:.2f}, BR = {pyr_stats['br_eff_mean']:.2f} ± {pyr_stats['br_eff_std']:.2f}\n")
        f.write(f"Effect Size Difference (FR vs BR): p = {pyr_stats['p_eff']}\n")
        
        if bp_stats_list:
            for bp_stat in bp_stats_list:
                f.write(f"{bp_stat['feature']}: Ctrl = {bp_stat['ctrl_mean']:.2f} ± {bp_stat['ctrl_std']:.2f}, DCZ = {bp_stat['dcz_mean']:.2f} ± {bp_stat['dcz_std']:.2f}, Change = {bp_stat['change_pct']:.2f}%, p = {bp_stat['p_value']}\n")
        
        if int_stats:
            f.write("\n=== L5 INTERNEURONS (PUFF) ===\n")
            f.write(f"n = {int_stats['n_neurons']}\n")
            f.write(f"FR: Ctrl = {int_stats['fr_ctrl_mean']:.2f} ± {int_stats['fr_ctrl_std']:.2f}, DCZ = {int_stats['fr_dcz_mean']:.2f} ± {int_stats['fr_dcz_std']:.2f}, Change = {int_stats['fr_change_pct']:.2f}%, p = {int_stats['p_FR']}\n")
            f.write(f"Effect Size: FR = {int_stats['fr_eff_mean']:.2f} ± {int_stats['fr_eff_std']:.2f}\n")
            f.write(f"Effect Size Difference (Inter vs other): p = {int_stats['p_eff']}\n")

    print(f"Analysis Complete. Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
