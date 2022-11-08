#%%
import torch
import pandas as pd

from utils import simulate_dag, Dist
from stein import SCORE

#%%
def generate(d, s0, N, noise_std = 1, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
    X, noise_var = teacher.sample(N)
    return X, adjacency


# Data generation paramters
graph_type = 'ER'
d = 10
s0 = 10
N = 1000

# X_s, adj = generate(d, s0, N, GP=True)

gt = pd.read_csv('~/Desktop/cliff-seer/results/ground_truths/gd_data_1.csv')
df = pd.read_csv('~/Downloads/data_1.csv')
X = torch.Tensor(df.values)

#%%
# SCORE hyper-parameters
eta_G = 0.01
eta_H = 0.01
cutoff = 1.0

A_SCORE, top_order_SCORE =  SCORE(X, eta_G, eta_H, cutoff)

print("-" * 10)
for si, source in enumerate(A_SCORE):
    for target in df.columns[source != 0]:
        print(f"{df.columns[si]} -> {target}")

# %%
# Find HPs
truth = "|".join([f"{s} -> {t}" for _, s, t, _ in gt.to_records()])

for eta_G in range(10, 1000, 1):
    for eta_H in range(10, 1000, 1):
        for cutoff in range(300, 10000, 1):
            A_SCORE, top_order_SCORE =  SCORE(X, eta_G/1000, eta_H/1000, cutoff/1000)
            
            graph = list()
            for si, source in enumerate(A_SCORE):
                for target in df.columns[source != 0]:
                    graph.append(f"{df.columns[si]} -> {target}")

            graph = "|".join(graph)
            print(graph)

            if truth == graph:
                print(f"GOT GT: eta_G = [{eta_G}], eta_H = [{eta_H}], cutoff = [{cutoff}]")
                break
