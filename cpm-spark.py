import re
import sys
import datetime;
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

cliques = cliques.keys();

# Save cliques
# cliques.keys().repartition(1).saveAsTextFile(k + "-cliques-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

cliques = cliques.collect();

upper_overlap_matrix_list = []


def is_k_1_adjacent(clique1, clique2):
    adjacency_nodes_counter = collections.Counter(clique1)
    community_counter = collections.Counter(clique2)
    overlap_counter = adjacency_nodes_counter & community_counter
    return len(overlap_counter) == int(k)


for i in range(len(cliques)):
    for j in range(i, len(cliques)):
        upper_overlap_matrix_list.append(is_k_1_adjacent(cliques[i], cliques[j]))



any_merge = True;
while any_merge:
    any_merge = False;

print(cliques.take(10))

## Finding K-1 Adjacent cliques and joining them to form communities
k_1_adjacent_nodes = cliques \
    .flatMap(lambda x: [(x[:j] + x[j + 1:]) for j in range(len(x))]) \
    .map(lambda x: tuple(x)) \
    .distinct()
k_1_adjacent_nodes_bc = sc.broadcast(k_1_adjacent_nodes.collect())


def is_k_1_overlap(adjacency_nodes, community):
    adjacency_nodes_counter = collections.Counter(adjacency_nodes)
    community_counter = collections.Counter(community)
    overlap_counter = adjacency_nodes_counter & community_counter
    return len(overlap_counter) == len(adjacency_nodes_counter)


final_communities = []
community_size = int(k)

while not cliques.isEmpty():
    communities = cliques \
        .flatMap(lambda x: [(nodes, x) for nodes in k_1_adjacent_nodes_bc.value if is_k_1_overlap(nodes, x)]) \
        .reduceByKey(lambda a, b: list(set(a) | set(b))) \
        .values() \
        .collect()

    # cliques = communities_bc.value.filter(lambda x: len(x) > community_size)
    # extracted_communities = communities_bc.value.filter(lambda x: len(x) == community_size)
    cliques = sc.parallelize(filter(lambda x: len(x) > community_size, communities))
    final_communities += filter(lambda x: len(x) == community_size, communities)
    community_size += 1

print(final_communities)

sc.stop()
