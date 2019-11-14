import re
import sys
import datetime;
import csv
import collections;
from pyspark import SparkConf, SparkContext

# Configure SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)

# Read edges
edges = sc.textFile(sys.argv[1])
k = sys.argv[2]

#### KC - Spark
### Inspired from:
### Dheekonda, R.S.R., 2017. Enumerating k-cliques in a large network using Apache Spark (Doctoral dissertation).

# Step 1 (Generate Adjacency list)
adjacencyList = edges \
    .map(lambda l: re.split(r' ', l)) \
    .map(lambda x: (int(x[0]), int(x[1]))) \
    .map(lambda x: (x[0], [x[1]]) if x[0] < x[1] else (x[1], [x[0]])) \
    .reduceByKey(lambda a, b: a + b)

# Step 2 (Broadcast adjacency list)
bc_adjacencyList = sc.broadcast(adjacencyList.collectAsMap())


# Step 3 (Finding K cliques)
def is_edge_exists(node, connected_nodes):
    return all(node in bc_adjacencyList.value.get(x) for x in connected_nodes)


cliques = adjacencyList \
    .map(lambda x: ([x[0]], x[1]))

for i in range(2, int(k) + 1):
    cliques = cliques \
        .flatMap(lambda x: [(x[0] + [x[1][j]], x[1][j + 1:]) for j in range(len(x[1]))]) \
        .filter(lambda x: is_edge_exists(x[0][-1], x[0][:-1]))

cliques = cliques \
    .keys() \
    .map(lambda x: tuple(x))

cliquesMap = {k: v for v, k in enumerate(cliques.collect())}
totalCliques = len(cliquesMap)

# Save clique Map
cliquesMapFile = open(k + "-cliques-map-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ".csv", "w")
w = csv.writer(cliquesMapFile)
for key, val in cliquesMap.items():
    w.writerow([val, key])
cliquesMapFile.close()

# Broadcast Cliques Map
cliquesMap_bc = sc.broadcast(cliquesMap)


# Find k-1 Adjacent cliques

def is_k_1_adjacent(clique1, clique2):
    adjacency_nodes_counter = collections.Counter(clique1)
    community_counter = collections.Counter(clique2)
    overlap_counter = adjacency_nodes_counter & community_counter
    return len(overlap_counter) == int(k)


k_1_adjacency_edges = cliques \
    .flatMap(lambda x: [((x[:j] + x[j + 1:]), [cliquesMap_bc.value[x]]) for j in range(len(x))]) \
    .reduceByKey(lambda a, b: a + b) \
    .values() \
    .flatMap(lambda x: [(v, x) for v in x]) \
    .reduceByKey(lambda a, b: a + b) \
    .flatMap(lambda x: [(x[0], v) for v in x[1] if x[0] > v])

print(k_1_adjacency_edges.take(10))


# Find connected components:
# Kiveris, R., Lattanzi, S., Mirrokni, V., Rastogi, V. and Vassilvitskii, S., 2014, November.
# Connected components in mapreduce and beyond. In Proceedings of the ACM Symposium on Cloud Computing (pp. 1-13). ACM.
# Vancouver

def small_star(edges_rdd):
    return edges_rdd \
        .map(lambda x: (x[0], [x[1]]) if x[0] > x[1] else (x[1], [x[0]])) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: (min(x[1]), x[1])) \
        .flatMap(lambda x: [(v, x[0]) for v in x[1]])


def large_star(edges_rdd):
    return edges_rdd \
        .flatMap(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])]) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: (min(x[1] + [x[0]]), x[1])) \
        .flatMap(lambda x: [(v, x[0]) for v in x[1] if v > x[0]])


s = k_1_adjacency_edges
t = large_star(small_star(s))
print(t.take(10))
while not s.subtract(t).isEmpty():
    s = t
    t = large_star(small_star(s))
    print(t.take(10))


sc.stop()
