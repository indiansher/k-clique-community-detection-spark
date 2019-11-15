import re
import sys
import datetime;
import csv
import time
import collections;
from pyspark import SparkConf, SparkContext

start = time.time()

# Configure SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)

# Read edges
edges = sc.textFile(sys.argv[1])
k = sys.argv[2]

#### KC - Spark
### Inspired from:
### Dheekonda, R.S.R., 2017. Enumerating k-cliques in a large network using Apache Spark (Doctoral dissertation).

print("Start KC Spark")

# Step 1 (Generate Adjacency list)
adjacencyList = edges \
    .map(lambda l: re.split(r'\s', l)) \
    .map(lambda x: (int(x[0]), int(x[1]))) \
    .map(lambda x: (x[0], [x[1]]) if x[0] < x[1] else (x[1], [x[0]])) \
    .reduceByKey(lambda a, b: a + b)

# Step 2 (Broadcast adjacency list)
bc_adjacencyMap = sc.broadcast(adjacencyList.collectAsMap())


# Step 3 (Finding K cliques)
def is_edge_exists(node, connected_nodes):
    return all(node in bc_adjacencyMap.value.get(x, []) for x in connected_nodes)


cliques = adjacencyList \
    .map(lambda x: ([x[0]], x[1]))

for i in range(2, int(k) + 1):
    cliques = cliques \
        .flatMap(lambda x: [(x[0] + [x[1][j]], x[1][j + 1:]) for j in range(len(x[1]))]) \
        .filter(lambda x: is_edge_exists(x[0][-1], x[0][:-1]))

cliques = cliques \
    .keys() \
    .map(lambda x: tuple(x))

cliquesList = cliques.collect()
cliquesMap = {k: v for v, k in enumerate(cliquesList)}
totalCliques = len(cliquesMap)

print("End KC Spark")

# Save clique Map
# cliquesMapFile = open(k + "-cliques-map-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ".csv", "w")
# w = csv.writer(cliquesMapFile)
# for key, val in cliquesMap.items():
#     w.writerow([val, key])
# cliquesMapFile.close()

print("Start finding adjacent k-1 cliques")

# Broadcast Cliques Map
cliquesMap_bc = sc.broadcast(cliquesMap)


# Find k-1 Adjacent cliques

def is_k_1_adjacent(clique1, clique2):
    adjacency_nodes_counter = collections.Counter(clique1)
    community_counter = collections.Counter(clique2)
    overlap_counter = adjacency_nodes_counter & community_counter
    return len(overlap_counter) == int(k)


k_1_adjacency_map = cliques \
    .flatMap(lambda x: [((x[:j] + x[j + 1:]), [cliquesMap_bc.value[x]]) for j in range(len(x))]) \
    .reduceByKey(lambda a, b: a + b) \
    .values() \
    .flatMap(lambda x: [(v, x) for v in x]) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: (x[0], [v for v in x[1] if x[0] < v])) \
    .collectAsMap()

print("End finding adjacent k-1 cliques")

print("Start connected Components")

# Find connected components (Using DFS)
visited = [False] * totalCliques


def dfs(node):
    visited_nodes = [node]
    visited[node] = True

    for v in k_1_adjacency_map[node]:
        if not visited[v]:
            visited_nodes.extend(dfs(v))

    return visited_nodes


# Run DFS
clique_communities = []
for v in range(totalCliques):
    if not visited[v]:
        clique_communities.append(dfs(v))

print("End connected components")

print("Start Replacing clique numbers with actual cliques")

# Replace clique numbers with actual cliques
communities = []
for clique_community in clique_communities:
    community = []
    for v in clique_community:
        community.extend(list((set(cliquesList[v]) - set(community))))
    communities.append(community)

print("End Replacing clique numbers with actual cliques")

end = time.time()
print("Execution Time: " + str(end - start) + " seconds")

print("Start output to file")

# Write to file
outputFileName = sys.argv[3]
# communitiesFile = open(k + "-communities-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ".txt", "w")
communitiesFile = open(outputFileName, "w")
for community in communities:
    communitiesFile.write("%s\n" % " ".join(map(str, community)))
communitiesFile.close()

print("End output to file")

sc.stop()
