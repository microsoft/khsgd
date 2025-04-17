# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.9.12 ('wg247-new-1')
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
# Add LaTeX packages (note: latex must be installed)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amssymb} \usepackage{amsmath}'
# Enable LaTeX rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] ='serif'
mpl.rcParams.update({"axes.grid" : True, "grid.color": "grey", "grid.linestyle": "--", "grid.linewidth": 0.1, "axes.grid.axis": "both", "grid.alpha": 0.25})

d_sorts = ["CD-GraB", "D-RR", "CD-OH-opt", "SBW"]
sorts_names = {"CD-GraB": "CD-GraB: Greedy", "D-RR": "RR", 
               "CD-OH-opt": "KH-SGD", "SBW": "CD-GraB: SBW"}
sorts_colors = {"CD-GraB": "tab:red", "D-RR": "tab:green", "CD-OH-opt": "tab:blue", "SBW": "tab:orange"}

fs = 20
fs12 = fs - 2
fs14 = fs - 2
fs10 = fs - 4
num_epochs = 50
def exp_maker(n, sorter, seed):
    return f"LR-hmda-{sorter}-lr-0.005-B-16-seed-{seed}"
exp_name = 'ft-epoch-pen-50'

# +
results = {s: [] for s in d_sorts}
times = {s: [] for s in d_sorts}
seconds = 'Seconds'
epochs = 'Epochs'

# Load results
for s in d_sorts:
    for seed in range(5):
        exp_details = exp_maker(4, s, seed)
        exp_folder = f"..{os.sep}..{os.sep}results{os.sep}{exp_name}{os.sep}{exp_details}"
        if not os.path.exists(exp_folder):
            continue
        r = torch.load(f"{exp_folder}{os.sep}results.pt", map_location='cpu')
        results[s].append(r)
        for rank in [0]:
            time_folder = f"{exp_folder}{os.sep}time{os.sep}"
            times[s].append(torch.load(f"{time_folder}time-{rank}.pt", map_location='cpu'))

# -

def down_sampling(len_sampling, data):
    start = 0
    end = len_sampling
    len_data = len(data)
    ret = []
    while end <= len_data:
        data_chunk = data[start:end]
        start += len_sampling
        end += len_sampling
        ret.append(torch.mean(torch.as_tensor(data_chunk)))
    return torch.as_tensor(ret)



def plot_res(ax, s, label, train_test, loss_acc, downsample=False, color=None):
    res = []
    for arr in results[s]:
        if downsample:
            if loss_acc == 'acc':
                res.append(down_sampling(2, 100 * torch.tensor(arr[train_test][loss_acc])))
            else:
                res.append(down_sampling(2, torch.tensor(arr[train_test][loss_acc])))
            step = 2
        else:
            if loss_acc == 'acc':
                res.append(100 * torch.tensor(arr[train_test][loss_acc]))
            else:
                res.append(torch.tensor(arr[train_test][loss_acc]))
            step = 1    
    end_pos = len(torch.tensor(results[s][0][train_test][loss_acc])) + 1
    print(res)
    res = torch.vstack(res).numpy()
    mean = res.mean(axis=0)
    std = res.std(axis=0)
    min = res.min(axis=0)
    max = res.max(axis=0)
    line = ax.plot(np.arange(1, end_pos, step), mean, label=label, markersize=5, color=color if color else None)
    ax.fill_between(
        np.arange(1, end_pos, step), (mean - std), (mean + std), alpha=0.1, color=color if color else None)
    return line[0]



# +
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=400, gridspec_kw=dict(wspace=0.25), sharex=True)



line1 = plot_res(axes[0], "D-RR", sorts_names["D-RR"], "train", 'loss', downsample=True, color=sorts_colors["D-RR"])
line2 = plot_res(axes[0], "CD-GraB", sorts_names["CD-GraB"], "train", 'loss', downsample=True, color=sorts_colors["CD-GraB"])
line3 = plot_res(axes[0], "SBW", sorts_names["SBW"], "train", 'loss', downsample=True, color=sorts_colors["SBW"])
line4 = plot_res(axes[0], "CD-OH-opt", sorts_names["CD-OH-opt"], "train", 'loss', downsample=True, color=sorts_colors["CD-OH-opt"])

# Turn off the upper line (top spine)
axes[0].spines['top'].set_visible(False)
# Turn off the right spine
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_position(('outward', 1))  # Move x-axis outward
axes[0].spines['left'].set_position(('outward', 1))    # Move y-axis outward

# # Add grid lines
# axes[0].grid(True)  # Adds default grid lines

# # Customize grid lines
# axes[0].grid(which='both', axis='both', color='gray', linestyle='--', linewidth=0.1, alpha=0.25)


# Put the legend left of the current axis
#axes[0].legend(fontsize=fs12, loc='lower left', handles=[line1, line2, line3, line4])
axes[0].set_title('Full Train Loss', fontsize=fs14)
axes[0].set_xlabel(epochs, fontsize=fs12)
axes[0].set_ylim(0.3345, 0.3395)

plot_res(axes[1], "D-RR", sorts_names["D-RR"], "test", 'acc', downsample=True, color=sorts_colors["D-RR"])
plot_res(axes[1], "SBW", sorts_names["SBW"], "test", 'acc', downsample=True, color=sorts_colors["SBW"])
plot_res(axes[1], "CD-OH-opt", sorts_names["CD-OH-opt"], "test", 'acc', downsample=True, color=sorts_colors["CD-OH-opt"])
plot_res(axes[1], "CD-GraB", sorts_names["CD-GraB"], "test", 'acc', downsample=True, color=sorts_colors["CD-GraB"])

# Turn off the upper line (top spine)
axes[1].spines['top'].set_visible(False)
# Turn off the right spine
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_position(('outward', 1))  # Move x-axis outward
axes[1].spines['left'].set_position(('outward', 1))    # Move y-axis outward


axes[1].set_title('Test Accuracy', fontsize=fs14)
axes[1].set_xlabel(epochs, fontsize=fs12)
axes[1].set_ylim(81.95, 82.45)
fig.tight_layout()
fig.savefig(f'..{os.sep}..{os.sep}notebooks{os.sep}LR-HMDA{os.sep}graph{os.sep}HMDA-epoch.png', bbox_inches="tight")
fig.savefig(f'..{os.sep}..{os.sep}notebooks{os.sep}LR-HMDA{os.sep}graph{os.sep}HMDA-epoch.pdf', bbox_inches="tight")

# -

def plot_time(ax, s, label, train_test, loss_acc, downsample=False, color=None):
    res = []
    for arr in results[s]:
        if downsample:
            if loss_acc == 'acc':
                res.append(down_sampling(2, 100 * torch.tensor(arr[train_test][loss_acc])))
            else:
                res.append(down_sampling(2, torch.tensor(arr[train_test][loss_acc])))
            step = 2
        else:
            if loss_acc == 'acc':
                res.append(100 * torch.tensor(arr[train_test][loss_acc]))
            else:
                res.append(torch.tensor(arr[train_test][loss_acc]))
            step = 1    
    res = torch.vstack(res).numpy()
    time_res = []
    for time_arr in times[s]:
        one_time = []
        for e in range(1, num_epochs + 1):
            one_time.append(sum(time_arr['time'][f'epoch-{e}']))
        one_time = torch.tensor(one_time)
        time_res.append(torch.cumsum(one_time, dim=0))
    time_res = torch.vstack(time_res).numpy()
    time_res = time_res.mean(axis=0)
    if downsample:
        time_res = down_sampling(2, time_res)
    mean = res.mean(axis=0)
    std = res.std(axis=0)
    ax.plot(time_res, mean, label=label, markersize=5, color=color if color else None)
    ax.fill_between(
        time_res, (mean - std), (mean + std), alpha=0.1, color=color if color else None)



# +
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=400, gridspec_kw=dict(wspace=0.25), sharex=True)

plot_time(axes[0], "SBW", sorts_names["SBW"], "train", 'loss', downsample=True, color=sorts_colors["SBW"])
plot_time(axes[0], "D-RR", sorts_names["D-RR"], "train", 'loss', downsample=True, color=sorts_colors["D-RR"])
plot_time(axes[0], "CD-OH-opt", sorts_names["CD-OH-opt"], "train", 'loss', downsample=True, color=sorts_colors["CD-OH-opt"])
plot_time(axes[0], "CD-GraB", sorts_names["CD-GraB"], "train", 'loss', downsample=True, color=sorts_colors["CD-GraB"])

# Turn off the upper line (top spine)
axes[0].spines['top'].set_visible(False)
# Turn off the right spine
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_position(('outward', 1))  # Move x-axis outward
axes[0].spines['left'].set_position(('outward', 1))    # Move y-axis outward

axes[0].legend(fontsize=fs10, loc='lower left')
axes[0].set_title('Full Train Loss', fontsize=fs14)
axes[0].set_xlabel(seconds, fontsize=fs12)
axes[0].set_xlim(0, 1200)
axes[0].set_ylim(0.3345, 0.3395)

plot_time(axes[1], "D-RR", sorts_names["D-RR"], "test", 'acc', downsample=True, color=sorts_colors["D-RR"])
plot_time(axes[1], "SBW", sorts_names["SBW"], "test", 'acc', downsample=True, color=sorts_colors["SBW"])
plot_time(axes[1], "CD-GraB", sorts_names["CD-GraB"], "test", 'acc', downsample=True, color=sorts_colors["CD-GraB"])
plot_time(axes[1], "CD-OH-opt", sorts_names["CD-OH-opt"], "test", 'acc', downsample=True, color=sorts_colors["CD-OH-opt"])

# # Add grid lines
# axes[0].grid(True)  # Adds default grid lines

# # Customize grid lines
# axes[0].grid(which='both', axis='both', color='gray', linestyle='--', linewidth=0.1, alpha=0.25)

# Turn off the upper line (top spine)
axes[1].spines['top'].set_visible(False)
# Turn off the right spine
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_position(('outward', 1))  # Move x-axis outward
axes[1].spines['left'].set_position(('outward', 1))    # Move y-axis outward

axes[1].set_title('Test Accuracy', fontsize=fs14)
axes[1].set_xlabel(seconds, fontsize=fs12)
axes[1].set_xlim(0, 1200)
axes[1].set_ylim(81.95, 82.45)

fig.tight_layout()
fig.savefig(f'..{os.sep}..{os.sep}notebooks{os.sep}LR-HMDA{os.sep}graph{os.sep}HMDA-second.png', bbox_inches="tight")
fig.savefig(f'..{os.sep}..{os.sep}notebooks{os.sep}LR-HMDA{os.sep}graph{os.sep}HMDA-second.pdf', bbox_inches="tight")
