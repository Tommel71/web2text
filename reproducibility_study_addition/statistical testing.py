import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind

sns.set(color_codes=True)


un_googletrends = pd.read_csv("data/results/googletrends/Unary_Results.csv")
struc_googletrends = pd.read_csv("data/results/googletrends/Structured_Results.csv")
alpha = 0.05
### cleaneval original split
split = "cleaneval_original_split"

paper_dict_orig = {"Accuracy": 0.84,
              "Precision": 0.88,
              "Recall": 0.85,
              "F1": 0.86}


struc = pd.read_csv("data/results/" + split + "/Structured_Results.csv")
un = pd.read_csv("data/results/" + split + "/Unary_Results.csv")
print(100*"|")
print(split)
print(100*"|")

from collections import defaultdict
pre_df = defaultdict(list)

for key in struc.columns:
    data_struc = struc[key]
    data_un = un[key]
    data_googletrends = struc_googletrends[key]
    obs = paper_dict_orig[key]
    n = len(data_struc)
    greater = (obs > data_struc).mean()
    smaller = (obs < data_struc).mean()
    print(key)

    fig, ax = plt.subplots()
    # create new data for plotting because there are outliers that mess up vis
    data_plot = data_struc.copy()
    data_plot = data_plot[data_plot > data_plot.quantile(0.001)]
    N, bins, patches = ax.hist(data_plot, linewidth=1, bins= 30)
    for i in range(len(patches)):
        patches[i].set_facecolor('b')

        if bins[i] > np.quantile(data_struc, 1-alpha/2):
            patches[i].set_facecolor('r')

        if bins[i] < np.quantile(data_struc, alpha/2):
            patches[i].set_facecolor('r')


    plt.axvline(obs, color='k', linestyle='dashed', linewidth=1)

    p = 2* min(greater, smaller)

    pre_df["paper observation p"].append(p)
    min_ylim, max_ylim = plt.ylim()
    plt.text(obs+ 0.001, max_ylim * 0.9, key + " in paper: " + str(obs))
    ax.set_title("Histogram of bootstrapped " + key, fontsize = 20)
    ax.set_xlabel(key)
    ax.set_ylabel("Count")
    plt.savefig("paper_observation_check/" +  split + "_" + key + ".jpg", dpi = 600)


    ##############
    # vs googletrends

    stat = ttest_ind(data_struc, data_googletrends)
    pre_df["CleanEval vs GT17 p"].append(stat[1])




    ##############
    # structured vs unary
    stat = ttest_rel(data_struc, data_un, alternative = "greater")
    df_plot = pd.DataFrame(data = [data_struc, data_un]).transpose()
    df_plot.columns = ["structured", "unary"]
    df_plot.plot()
    plt.savefig("structured_vs_unary/" + split + "_" + key + ".jpg", dpi = 600)
    pre_df["structured vs unary p"].append(stat[1])



df = pd.DataFrame(pre_df)
df.index = struc.columns
print(df)
df.to_csv("cleanevalsplit_p_vals.csv")

struc.columns = [a  + " CleanEval" for a in struc.columns]
struc_googletrends.columns = [a  + " GT17" for a in struc_googletrends.columns]
all = pd.concat([struc, struc_googletrends], axis = 1)
all = all[sorted(all.columns)]
all.plot.box(vert=False)
plt.tight_layout()
plt.savefig("boxplots.jpg", dpi =600)



##########################################

### Web2Text split
split = "cleaneval_web2text_split"
paper_dict_orig = {"Accuracy": 0.86,
              "Precision": 0.87,
              "Recall": 0.90,
              "F1": 0.88}



struc = pd.read_csv("data/results/" + split + "/Structured_Results.csv")
un = pd.read_csv("data/results/" + split + "/Unary_Results.csv")
print(100*"|")
print(split)
print(100*"|")

from collections import defaultdict
pre_df = defaultdict(list)

for key in struc.columns:
    data_struc = struc[key]
    data_un = un[key]
    obs = paper_dict_orig[key]
    n = len(data_struc)
    greater = (obs > data_struc).mean()
    smaller = (obs < data_struc).mean()
    print(key)

    fig, ax = plt.subplots()
    # create new data for plotting because there are outliers that mess up vis
    data_plot = data_struc.copy()
    data_plot = data_plot[data_plot > data_plot.quantile(0.001)]
    N, bins, patches = ax.hist(data_plot, linewidth=1, bins= 30)
    for i in range(len(patches)):
        patches[i].set_facecolor('b')

        if bins[i] > np.quantile(data_struc, 1-alpha/2):
            patches[i].set_facecolor('r')

        if bins[i] < np.quantile(data_struc, alpha/2):
            patches[i].set_facecolor('r')


    plt.axvline(obs, color='k', linestyle='dashed', linewidth=1)

    p = 2* min(greater, smaller)

    pre_df["paper observation p"].append(p)
    min_ylim, max_ylim = plt.ylim()
    plt.text(obs+ 0.001, max_ylim * 0.9, key + " in paper: " + str(obs))
    ax.set_title("Histogram of bootstrapped " + key, fontsize = 20)
    ax.set_xlabel(key)
    ax.set_ylabel("Count")
    plt.savefig("paper_observation_check/" +  split + "_" + key + ".jpg", dpi = 600)


df = pd.DataFrame(pre_df)
df.index = struc.columns
print(df)
df.to_csv("web2textsplit_p_vals.csv")
