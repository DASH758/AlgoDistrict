from gerrychain import Graph, Election, updaters, Partition, GeographicPartition, constraints, MarkovChain
from gerrychain.updaters import cut_edges
from gerrychain.metrics.compactness import polsby_popper
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.tree import recursive_tree_part
from gerrychain.accept import always_accept
from gerrychain.constraints import single_flip_contiguous
from gerrychain.optimization import Gingleator, SingleMetricOptimizer
from gerrychain.metrics.partisan import efficiency_gap, mean_median, partisan_bias
from functools import partial
import matplotlib.pyplot as plt
import gerrytools.plotting as gplot
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import csv
import pickle

NUMDISTRICTS = 38
p1 = gpd.read_file("tx_vtd_2020_bound.shx")
p1 = p1.drop(columns=['STATEFP20', 'COUNTYFP20', 'VTDST20', 'VTDI20', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'FUNCSTAT20', 'NAME20', 'INTPTLAT20', 'INTPTLON20'])
p1 = p1.sort_values(by='GEOID20')
p1['AREA'] = p1['ALAND20'] + p1['AWATER20']
p1 = p1.drop(columns=['ALAND20', 'AWATER20'])

p2 = pd.read_csv("tx-precinct-data.csv")
precincts = p1.merge(p2, on='GEOID20')
precincts.to_crs(epsg=5070, inplace=True)


precincts.plot()
plt.show()


graph = Graph.from_json("tx_crs_corrected_data.json")
graph.to_json("tx_crs_corrected_data.json")

election_names = ["COMP16-20"] 
election_columns = [["E_16-20_COMP_Dem", "E_16-20_COMP_Rep"]]
pop_col = "ADJPOP"

myupdaters = {
    "population": updaters.Tally(pop_col, alias="population"),
    "cut_edges": cut_edges,
    "polsby-popper": polsby_popper
}

elections = [
    Election(name,{"Democratic": dem, "Republican": rep})
    for name, (dem, rep) in zip(election_names, election_columns)
]
election_updaters = {election.name: election for election in elections}
myupdaters.update(election_updaters)

total_population = sum([graph.nodes[n][pop_col] for n in graph.nodes])
assignment = recursive_tree_part(
    graph, 
    range(NUMDISTRICTS),
    total_population/NUMDISTRICTS,
    pop_col, 
    0.05
)

initial_partition = GeographicPartition(graph, assignment, myupdaters)

initial_partition.plot(precincts)
plt.show()

myconstraints = [
    constraints.within_percent_of_ideal_population(initial_partition, 0.05) 
]

myproposal = partial(
    recom,
    pop_col=pop_col,
    pop_target=total_population/NUMDISTRICTS,
    epsilon=0.05, 
    node_repeats=2, 
)

heur = lambda x: 20*(abs(x["COMP16-20"].efficiency_gap())+abs(x["COMP16-20"].mean_median()+abs(x["COMP16-20"].partisan_bias())))/3 + (NUMDISTRICTS-sum(x["polsby-popper"].values()))/len(x)

optimizer = SingleMetricOptimizer(
    proposal=myproposal,
    constraints=myconstraints,
    initial_state=initial_partition,
    optimization_metric=heur,
    maximize=False
)
total_steps = 10000

# Short Bursts
min_scores_sb = np.zeros(total_steps)
for i, part in enumerate(optimizer.short_bursts(5, 2000, with_progress_bar=True)):
    min_scores_sb[i] = optimizer.best_score
print(optimizer.best_score)
optimizer.best_part.plot(precincts)
plt.show()

data = [["GEOID20", "District"]]
for n in list(optimizer.best_part.graph.nodes):
    data.append([optimizer.best_part.graph.nodes[n]["GEOID20"], optimizer.best_part.assignment[n]])
with open("final.csv",'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Simulated Annealing
min_scores_anneal = np.zeros(total_steps)
for i, part in enumerate(
    optimizer.simulated_annealing(
        total_steps,
        optimizer.jumpcycle_beta_function(200, 800),
        beta_magnitude=1,
        with_progress_bar=True
    )
):
    min_scores_anneal[i] = optimizer.best_score

# Tilted Runs
min_scores_tilt = np.zeros(total_steps)
for i, part in enumerate(optimizer.tilted_run(total_steps, p=0.125, with_progress_bar=True)):
    min_scores_tilt[i] = optimizer.best_score

fig, ax = plt.subplots(figsize=(12,6))
plt.plot(min_scores_sb, label="Short Bursts")
plt.plot(min_scores_anneal, label="Simulated Annealing")
plt.plot(min_scores_tilt, label="Tilted Run")
plt.xlabel("Steps", fontsize=20)
plt.ylabel("Efficiency Gap", fontsize=20)
plt.legend()
plt.show()



