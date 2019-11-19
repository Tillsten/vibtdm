import matplotlib.pyplot as plt
import numpy as np



from cycler import cycler

def vibtdm_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (3.5, 3)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 9
    custom_cycler = cycler('color', ['k', 'r', 'b', 'g', 'xkcd:navy'])
    plt.rcParams['axes.prop_cycle'] = custom_cycler
    plt.rcParams['legend.labelspacing'] = 0.2




def plot_vibs(vibs_dict: dict, y_off: float = 0.0, ax=None, res_offset:float=0):
    if ax is None:
        ax = plt.gca()
    res_offset = 0
    # print(vibs_dict)
    for key, vib in vibs_dict.items():
        if key in ('Am1', 'Am1nr', 'Am1nr0'):
            continue
        ax.hlines(y_off + res_offset, *vib.freq_range)
        mid = (vib.freq_range[0] + vib.freq_range[1]) / 2.0
        angle = vib.angle if vib.angle < 90 else 180 - vib.angle
        #print(angle, vib)
        ax.text(mid,
                y_off + res_offset,
                '%.1f°' % angle,
                ha="center",
                va="bottom")
        ax.text(mid, y_off - res_offset, vib.name, ha="center", va="top")
        res_offset = res_offset + 0.3
    return res_offset
