import numpy as np
import pickle
import itertools
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("ggplot")

# this is all quite hacky and not good code 

# load embedding data, created e.g. in google colab
with open("articles_tags.pkl", "rb") as f:
    articles_tags = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

topic_dict = {"Israel_Hamas_": 0, "US_elections_": 1,
             "Bangladesh_": 2}
topic_index = ["Israel-Palestine", "US elections (Harris)", "Bangladesh (Yunus)"]
language_dict = {"English/EN_": 0, "German/DE_": 1, "Spanish/ESP_": 2}
language_index = ["English", "German", "Spanish"]

mapdict = lambda dic: lambda ls: map(lambda x: dic[x], ls)

# repeat once because we have generated and baseline tags 
topics = np.array(list(mapdict(topic_dict) ([at["topic"] for at in articles_tags] * 2)))
language = np.array(list(mapdict(language_dict) ([at["lang"] for at in articles_tags] *2)))
generated = np.array([0] * 27 + [1] * 27)

umap_2d = umap.UMAP(n_components=2, random_state=27)
proj_2d = umap_2d.fit_transform(embeddings)

markers = ["x", "o"]
topic_labels = ['Israel-Palestine', 'Kamala Harris', "Muhammad Yunus"]
language_labels = ['English', 'German', 'Spanish']
gen_labels = ['Data tags', 'Generated Tags'] 
size = 30
colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 6))
fig.supxlabel("Embedding dimension 1", fontsize=18)
fig.supylabel("Embedding dimension 2", fontsize=18)

for i in range(2): 
    idx = np.array(range(0 + i*27, 27*(i+1)))
    proj_subset = proj_2d[idx]
    topic_subset = topics[idx]
    scatter = axs[0].scatter(proj_subset[:, 0], proj_subset[:, 1], c=colors[topic_subset], marker=markers[i], 
                             s=size, label=gen_labels[i])

legend_elements = [Line2D([0], [0], marker='s', linestyle="None", color="w", label=topic_labels[i], markerfacecolor=colors[i], markersize=15) for i in range(3)]
axs[0].legend(handles=legend_elements, loc="lower left", fontsize=15, title="Topic", title_fontsize=15)


for i in range(2): 
    idx = np.array(range(0 + i*27, 27*(i+1)))
    proj_subset = proj_2d[idx]
    lang_subset = language[idx]
    scatter = axs[1].scatter(proj_subset[:, 0], proj_subset[:, 1], c=colors[lang_subset + 3], marker=markers[i],
                             s=size, label=gen_labels[i])

legend_elements = [Line2D([0], [0], marker='s', linestyle="None", color="w", label=language_labels[i], markerfacecolor=colors[i + 3], markersize=15) for i in range(3)]
axs[1].legend(handles=legend_elements, loc="lower left", fontsize=15, title = "Language", title_fontsize=15)


handles = [Line2D([0], [0], marker="x", color="black", linestyle="None", label=gen_labels[0], markerfacecolor="black", markersize=15),
           Line2D([0], [0], marker="o", color="black", linestyle="None", label=gen_labels[1], markerfacecolor="black", markersize=15)]
fig.legend(handles=handles, loc="upper center", fontsize=15, bbox_to_anchor=(0.5, 1.015))

# import tikzplotlib
fig.savefig("embeddings_plot.pdf")

# plt.show()