import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import emcee
import batman
import corner
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocess as mp
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde

def alpha_plotting(info, nonlinear_results = None, linear_results = None, vanilla_results = None, which=("vanilla", "linear", "nonlinear")):
    datasets = []
    if "vanilla" in which and vanilla_results is not None:
        datasets.append(("vanilla", vanilla_results["alpha_samps_van"], 
                         vanilla_results["best_alpha_sample_van"]))

    if "linear" in which and linear_results is not None:
        datasets.append(("linear", linear_results["linear_alpha_samples"], 
                         linear_results["linear_best_alpha_samp"]))

    if "nonlinear" in which and nonlinear_results is not None:
        datasets.append(("nonlinear", nonlinear_results["nonlinear_alpha_samples"], 
                         nonlinear_results["nonlinear_best_alpha_samp"]))
        

   

    
    raw_labels = info["alpha_names"]
    labels = [r"$\rho_\star$" if name == "ln_rho_s" else name for name in raw_labels]
    colors = ["C0", "C1", "C2"]

    ranges = []
    for i in range(len(labels)):
        x = np.concatenate([arr[:, i] for _, arr, _ in datasets])
        lo, hi = np.percentile(x, [2.5, 97.5])
        pad = 0.05 * (hi - lo) if hi > lo else 1e-6
        ranges.append((lo - pad, hi + pad)) 

    fig = None
    for i, (variation, samples, truths_vec) in enumerate(datasets):
        kwargs = dict(labels=labels, color=colors[i], bins=30, range=ranges)
        kwargs["truths"] = truths_vec
        if fig is None:
            fig = corner.corner(
                samples, show_titles=False, title_fmt=".2f",
                hist_kwargs={"alpha": 0.6, "density" : True},
                contour_kwargs={"colors": [colors[i]]},
                **kwargs
            )
        else:
            fig = corner.corner(
                samples, fig=fig, plot_datapoints=False, show_titles=False,
                hist_kwargs={"alpha": 0.4, "density" : True},
                contour_kwargs={"colors": [colors[i]]},
                **kwargs
            )
    if fig is not None and fig.axes:
        ax0 = fig.axes[0]
        handles = [Line2D([0],[0], color=colors[i], lw=2, label=name) for i,(name,_,_) in enumerate(datasets)]
        ax0.legend(handles=handles, frameon=False, fontsize="small")
    
    plt.tight_layout()
    plt.show()  

    return

def beta_plotting(info, nonlinear_results = None, linear_results = None, vanilla_results = None, which=("vanilla", "linear", "nonlinear")):
    
    if info is None:
        raise ValueError("'info' dictionary must be provided with 'ncurves' and 'beta_num'.")

    ncurves = info.get("ncurves")
    beta_num = info.get("beta_num")
    if ncurves is None or beta_num is None:
        raise ValueError("'info' must contain 'ncurves' and 'beta_num'.")

    # Prepare colours for different algorithms
    colors = {"Vanilla": "C0", "linear": "C1", "nonlinear": "C2"}

    figures = []
    for i in range(ncurves):
        # Collect datasets for the current curve
        datasets = []
        # Vanilla betas are stored as a 2D array of shape (n_samples, ncurves*beta_num)
        if vanilla_results is not None:
            beta_samps_van = vanilla_results.get("beta_samps_van")
            if beta_samps_van is None:
                beta_samps_van = vanilla_results.get("beta_samps")
            if beta_samps_van is not None and beta_samps_van.size > 0:
                start = i * beta_num
                end = (i + 1) * beta_num
                datasets.append(("Vanilla", beta_samps_van[:, start:end]))
        # linear‑profile betas are shape (n_samples, beta_num, ncurves)
        if linear_results is not None:
            beta_linear = linear_results.get("linear_betas")
            if beta_linear is not None and beta_linear.size > 0:
                datasets.append(("linear", beta_linear[:, :, i]))
        # nonlinear/profile betas are shape (n_samples, beta_num, ncurves)
        if nonlinear_results is not None:
            beta_nonlinear = nonlinear_results.get("nonlinear_betas")
            if beta_nonlinear is not None and beta_nonlinear.size > 0:
                datasets.append(("nonlinear", beta_nonlinear[:, :, i]))


        ranges = []
        for dim in range(beta_num):
            combined = np.concatenate([data[:, dim] for _, data in datasets])
            lo, hi = np.percentile(combined, [2.5, 97.5])
            if lo == hi:
                pad = abs(lo) * 1e-3 if lo != 0 else 1e-6
                lo -= pad
                hi += pad
            pad = 0.05 * (hi - lo)
            ranges.append((lo - pad, hi + pad))

        # Generate corner plot for this curve
        fig = None
        labels = [f"β{j}" for j in range(beta_num)]
        for idx, (name, data) in enumerate(datasets):
            colour = colors.get(name, f"C{idx}")
            hist_alpha = 0.6 if fig is None else 0.4
            if fig is None:
                fig = corner.corner(
                    data,
                    labels=labels,
                    color=colour,
                    bins=30,
                    range=ranges,
                    hist_kwargs={"alpha": hist_alpha},
                    contour_kwargs={"colors": [colour]},
                    show_titles=True,
                    title_fmt=".2f",
                )
            else:
                corner.corner(
                    data,
                    fig=fig,
                    labels=labels,
                    color=colour,
                    bins=30,
                    range=ranges,
                    hist_kwargs={"alpha": hist_alpha},
                    contour_kwargs={"colors": [colour]},
                    plot_datapoints=False,
                    show_titles=False,
                )

        # Add legend and title
        if fig is not None:
            ax0 = fig.axes[0]
            handles = [Line2D([0], [0], color=colors.get(name, f"C{j}"), lw=2, label=name)
                       for j, (name, _) in enumerate(datasets)]
            ax0.legend(handles=handles, frameon=False, fontsize="small")
            fig.suptitle(f"β parameters for curve {i}", y=1.02, fontsize="x-large")
            figures.append(fig)

        plt.show()
    return 

def psi_plotting(info, nonlinear_results = None, linear_results = None, vanilla_results = None, which=("vanilla", "linear", "nonlinear")):
    datasets = []
    if "vanilla" in which and vanilla_results is not None:
        datasets.append(("vanilla", vanilla_results["psi_samps_van"], 
                         "C0"))

    if "linear" in which and linear_results is not None:
        datasets.append(("linear", linear_results["linear_psi_samps"], 
                         "C1"))

    if "nonlinear" in which and nonlinear_results is not None:
        datasets.append(("nonlinear", nonlinear_results["nonlinear_psi_samps"], 
                         "C2"))


    ncurves = info["ncurves"]


    psi_name = info.get("psi_name", "psi")
    labels   = [f"{psi_name}_{i+1}" for i in range(ncurves)]

    ranges = []
    for j in range(ncurves):
        col = np.concatenate([d[:, j] for _, d, _ in datasets], axis=0)
        lo, hi = np.percentile(col, [2.5, 97.5])
        width  = hi - lo
        pad    = 0.05 * width if width > 0 else 1e-3
        ranges.append((lo - pad, hi + pad))

    fig = None
    for k, (name, arr, color) in enumerate(datasets):
        # Labels only on the first call; `corner` ignores duplicates anyway
        fig = corner.corner(
            arr,
            labels=labels if k == 0 else None,
            range=ranges,
            bins=30,
            color=color,
            plot_datapoints=True,
            fill_contours=False,
            hist_kwargs={"alpha": 0.55, "density": True},
            fig=fig,
        )

    if fig is not None and fig.axes:
        ax0 = fig.axes[0]
        handles = [Line2D([0], [0], color=color, lw=2, label=name)
                   for name, _, color in datasets]
        ax0.legend(handles=handles, frameon=False, fontsize="small")

    fig.tight_layout()

    plt.show()

    return 

