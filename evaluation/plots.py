import numpy as np 
import matplotlib.pyplot as plt 

german_original = np.random.normal(size = (10, 384))
german_generated = np.random.normal(loc = 1, size= (10, 384))
english_original = np.random.normal(loc = 3, size = (10, 384))
english_generated = np.random.normal(loc = 4, size= (10, 384))
spanish_original = np.random.normal(loc = -1, size = (10, 384))
spanish_generated = np.random.normal(loc = -2, size= (10, 384))

def cosine_sim(x, y): 
    return x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))

def all_sims(xs, ys): 
    res = np.zeros(shape = (10))
    for i in range(0, 10): 
        res[i] = cosine_sim(xs[i], ys[i])
    return res

german_sims = all_sims(german_original, german_generated)
english_sims = all_sims(english_original, english_generated)
spanish_sims = all_sims(spanish_original, spanish_generated)

names = ["DE"] * 10 + ["EN"] * 10 + ["ESP"] * 10
fig, ax = plt.subplots(figsize = 2, 4) 
ax.scatter(names, np.concatenate([german_sims, english_sims, spanish_sims]))

