from multiprocessing import Pool 
from tqdm import tqdm 
from collections import defaultdict 
import networkx as nx 
import pandas as pd 


develop_data = "/Users/zyang/Documents/VSCode/DeepSpin/develop.data"


def get_node_pair_iterator(node_lst): 
    # yield ('hello world', 'hello my world')
    for i in range(len(node_lst)-1): 
        for j in range(i+1, len(node_lst)): 
            yield (node_lst[i], node_lst[j])

overlap_dict = defaultdict(int)

i = 0 
with open(develop_data, 'r') as fin: 
    lines = fin.readlines()


G = nx.Graph()
for line in lines: 
    node1, node2, _ = line.split('\t')
    G.add_edge(node1, node2, connect_type="hard")

node_lst = list(G.nodes())

def build_new_edge(node1, node2): 
    global i 
    global overlap_dict
    # node1, node2 = node_pair
    if node1 in G.neighbors(node2): 
        return
    overlap_set = set(node1.split(' ')).intersection(set(node2.split(' ')))
    for token in overlap_set: 
        overlap_dict[token] += 1 
    if overlap_set: 
        G.add_edge(node1, node2, connect_type="soft")
        i += 1 
        if i % 100000 == 0: 
            print(f'{i//100000} millions mapped')
# build extra node connection 


# for node_pair in tqdm(get_node_pair_iterator(node_lst), total=(len(node_lst))*(len(node_lst)-1)//2):
#     build_new_edge(node_pair)


if __name__ == '__main__':


    with Pool(50) as p: 
        p.starmap(build_new_edge, get_node_pair_iterator(node_lst))

    df = pd.DataFrame.from_dict(overlap_dict, orient='index', columns=['count'])
    df.reset_index().to_csv("/Users/zyang/Documents/VSCode/DeepSpin/overlap_count.csv", index=False)