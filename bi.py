import heapq
import random
import networkx as nx

visited_nodes = set()


def request_generator(node_num, req_num, max_band):
    random.seed(1)
    req_list = []
    for i in range(0, req_num):
        src = random.randint(0, node_num)
        dst = random.randint(0, node_num)
        while dst == src:
            dst = random.randint(0, node_num)
        band = random.randint(0, max_band)
        req_list.append((src, dst, band))
    return req_list


def route_exist(bi_list, src, dst):
    for i in bi_list:
        if src in i and dst in i:
            return i
    return None


def construct_single_bi_from_node_cpu(G, beta, x):
    global visited_nodes
    visited_nodes = set()

    return construct_bi_from_node_cpu(G, beta, x)


def construct_single_bi_from_edge_band(G, beta, x):
    global visited_nodes
    visited_nodes = set()

    return construct_bi_from_edge_band(G, beta, x)


def construct_bi_cpu(G, beta):
    global visited_nodes
    bi_list = []
    visited_nodes = set()
    for v in G.nodes():
        if v in visited_nodes:
            continue
        visited_nodes.add(v)
        bi = construct_bi_from_node_cpu(G, beta, v)
        if len(bi) > 0:
            bi_list.append(bi)
        else:
            bi_list.append(set())
    return bi_list


def construct_bi_bandwidth(G, beta):
    #print(G, beta)

    global visited_nodes
    bi_list = []
    visited_nodes = set()
    for v in G.nodes():
        if v in visited_nodes:
            continue
        visited_nodes.add(v)
        #        print "V:", visited_nodes
        bi = construct_bi_from_edge_band(G, beta, v)
        bi_list.append(bi)
    return bi_list


def construct_bi_from_node_cpu(G, beta, x):
    global visited_nodes
    bi = set()  # result bi set

    s = set()  # set for nodes

    visited_nodes.add(x)
    if G.nodes[x]["cpu"] >= beta:
        s.add(x)

    while s:
        n = s.pop()
        bi.add(n)
        for w, e in G[n].items():
            if G.nodes[w]["cpu"] >= beta and w not in visited_nodes:
                s.add(w)
            # In node-cpu-based bi, given beta, a node is either in one or
            # none of the bis.
            visited_nodes.add(w)
    return bi


def construct_bi_from_edge_band(G, beta, x):
    global visited_nodes
    bi = {x}  # result bi set
    s = set()  # set for links
    for w, e in G[x].items():
        if e['weight'] >= beta and w not in visited_nodes:
            s.add((x, w, e['weight']))
    while s:
        #        print s
        l = s.pop()
        if l[1] not in bi:
            visited_nodes.add(l[1])
            #           print "#", l[1]
            bi.add(l[1])
            for w, e in G[l[1]].items():
                if e['weight'] >= beta and l[0] != w and w not in visited_nodes:
                    s.add((l[1], w, e['weight']))

    return bi


def bfs_optimize_minimum(G, source, target=None, cutoff=None, weight='weight', lowerbound=0):
    if source == target:
        raise Exception("Not reach")
        return {source: 0}, {source: [source]}
    dist = {}  # dictionary of final distances
    paths = {source: []}  # dictionary of paths
    seen = {source: 0}
    seen_min = {source: float("inf")}
    fringe = []  # use heapq with (distance,label) tuples
    heapq.heappush(fringe, (0, source))
    while fringe:
        (d, v) = heapq.heappop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        edata = iter(G[v].items())

        for w, edgedata in edata:
            tmp = edgedata.get(weight, 1)
            if tmp < lowerbound:
                continue
            vw_dist = dist[v] + 1
            if cutoff is not None:
                if vw_dist > cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif (w not in seen) or (vw_dist < seen[w]) or (vw_dist == seen[w] and seen_min[w] < min(seen_min[v], tmp)):
                seen[w] = vw_dist
                seen_min[w] = min(seen_min[v], tmp)
                heapq.heappush(fringe, (vw_dist, w))
                paths[w] = paths[v] + [w]
    return dist, paths


def find_maximal_path_minimal_edge(G, source, target=None, weight='weight', upperbound=0):
    lb = 0
    ub = upperbound

    while ub - lb > 1:
        mid = (lb + ub) // 2
        dist, paths = bfs_optimize_minimum(G, source=source, target=target, weight=weight, lowerbound=mid)
        if target not in dist:
            ub = mid
        else:
            lb = mid

    return lb


# deprecated
def single_source_dijkstra_prune_edges(G, source, target=None, cutoff=None, weight='weight', lowerbound=0):
    if source == target:
        return ({source: 0}, {source: [source]})
    dist = {}  # dictionary of final distances
    paths = {source: [source]}  # dictionary of paths
    seen = {source: 0}
    fringe = []  # use heapq with (distance,label) tuples
    heapq.heappush(fringe, (0, source))
    while fringe:
        (d, v) = heapq.heappop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        edata = iter(G[v].items())

        for w, edgedata in edata:
            tmp = edgedata.get(weight, 1)
            if tmp < lowerbound:
                continue
            vw_dist = dist[v] + tmp
            if cutoff is not None:
                if vw_dist > cutoff:
                    continue
            if w in dist:
                if vw_dist < dist[w]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif w not in seen or vw_dist < seen[w]:
                seen[w] = vw_dist
                heapq.heappush(fringe, (vw_dist, w))
                paths[w] = paths[v] + [w]
    return (dist, paths)
