import copy
from enum import Enum, unique
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import bi
import consts
import math


@unique
class NetworkType(Enum):
    Substrate = 0
    Virtual = 1


@unique
class NetworkConstructorType(Enum):
    Random = 0
    StdIn = 1


@unique
class BIType(Enum):
    NodeCpu = 0
    LinkBand = 1


# TODO: add a subclass denoting cpu/band bis if needed.
class Network:
    def __init__(self, Graph, node_bound, cpu_bound, edge_bound, bandwidth_bound, network_type):
        self.Graph = Graph
        self.node_bound = node_bound
        self.cpu_bound = cpu_bound
        self.edge_bound = edge_bound
        self.bandwidth_bound = bandwidth_bound
        self.network_type = network_type

        self.node_cpu_beta_mp = {}
        self.node_band_beta_mp = {}
        self.bi_list = []

    def sum_cpu_band(self):
        rev = 0
        for node in self.Graph.nodes:
            rev += self.Graph.nodes[node]["cpu"]
        for u, v, weight in self.Graph.edges.data("weight"):
            rev += weight

        return rev

    def sum_band(self):
        band = 0
        for u, v, weight in self.Graph.edges.data("weight"):
            band += weight

        return band

    def plot(self):
        plt.title(
            f"{self.network_type} Network, node_num={self.Graph.number_of_nodes()}, cpu_bound={self.cpu_bound}\n" +
            f"edge_num={self.Graph.number_of_edges()}, band_bound={self.bandwidth_bound}")
        plot_graph_raw(self.Graph)

    def calculate_all_nodes_beta_bis(self):
        steps = 5
        self.bi_list = self.get_all_bis(steps=steps)
        self.node_cpu_beta_mp = {}
        self.node_band_beta_mp = {}
        for bi in self.bi_list:
            for node in bi.Graph.nodes:
                if node not in self.node_cpu_beta_mp:
                    self.node_cpu_beta_mp[node] = {}
                if node not in self.node_band_beta_mp:
                    self.node_band_beta_mp[node] = {}

                if bi.bi_type == BIType.NodeCpu:
                    self.node_cpu_beta_mp[node][bi.step] = bi
                else:
                    self.node_band_beta_mp[node][bi.step] = bi

        for node, mp in self.node_cpu_beta_mp.items():
            for stride in range(0, steps):
                if stride not in mp:
                    mp[stride] = self.build_network_from_bi_model(set(), BIType.NodeCpu, stride * self.cpu_bound / steps, stride)
                    self.node_cpu_beta_mp[node] = mp

        #print(self.node_cpu_beta_mp)
        #print(self.node_band_beta_mp)

    def get_all_bis(self, steps=5):
        if steps <= 0:
            steps = 5
        bi_list = []

        cpu_beta = 0
        cpu_stride = self.cpu_bound / steps
        for i in range(0, steps):
            bi_list += self.build_networks_from_bi_models(bi.construct_bi_cpu(self.Graph, cpu_beta), BIType.NodeCpu,
                                                          cpu_beta, i)
            cpu_beta += cpu_stride

        band_beta = 0
        band_stride = self.bandwidth_bound / steps
        for i in range(0, steps):
            bi_list += self.build_networks_from_bi_models(bi.construct_bi_bandwidth(self.Graph, band_beta),
                                                          BIType.LinkBand,
                                                          band_beta, i)
            band_beta += band_stride

        return bi_list

    def build_network_from_bi_model(self, bm, bi_type, beta, step):
        return BINetwork(nx.subgraph(self.Graph, bm), self.node_bound, self.cpu_bound, self.edge_bound,
                         self.bandwidth_bound,
                         self.network_type, bi_type, beta, step)

    def build_networks_from_bi_models(self, bms, bi_type, beta, step):
        network_list = []
        for bm in bms:
            network_list.append(self.build_network_from_bi_model(bm, bi_type, beta, step))
        return network_list

    def get_bi_from_node_cpu(self, beta, x):
        node_set = bi.construct_single_bi_from_node_cpu(self.Graph, beta, x)
        if node_set is None:
            node_set = set()

        return self.build_network_from_bi_model(node_set, BIType.NodeCpu, beta)

    def get_bi_from_edge_band(self, beta, x):
        node_set = bi.construct_single_bi_from_edge_band(self.Graph, beta, x)
        if node_set is None:
            node_set = set()

        return self.build_network_from_bi_model(node_set, BIType.LinkBand, beta)

    # TODO: test
    def after(self, mapping_solution):
        result_graph = copy.deepcopy(self.Graph)
        #print("NODE MAP", mapping_solution.node_map)
        # handle node mapping
        for v_node, s_node in mapping_solution.node_map.items():
            # check if nodes exists
            if not mapping_solution.VN.Graph.has_node(v_node):
                raise Exception(f"virtual node {v_node} not in VN")
            if not result_graph.has_node(s_node):
                raise Exception(f"substrate node {s_node} not in SN")

            # check if substrate node has enough cpu
            if result_graph.nodes[s_node]["cpu"] < mapping_solution.VN.Graph.nodes[v_node]["cpu"]:
                raise Exception(f"substrate node {s_node} has cpu {result_graph.nodes[s_node]['cpu']}, "
                      f"but virtual node {v_node} requires {mapping_solution.VN.Graph.nodes[v_node]['cpu']} cpu")

            result_graph.nodes[s_node]["cpu"] -= mapping_solution.VN.Graph.nodes[v_node]['cpu']

        for index in range(len(mapping_solution.virtual_links)):
            v_edge = mapping_solution.virtual_links[index]
            s_path = mapping_solution.mapped_links[index]

            if not mapping_solution.VN.Graph.has_edge(v_edge['from'], v_edge['to']):
                raise Exception(f"virtual edge ({v_edge['from']}, {v_edge['to']} not in VN)")

            for s_edge in s_path:
                if not result_graph.has_edge(s_edge[0], s_edge[1]):
                    raise Exception(f"substrate edge ({s_edge[0]}, {s_edge[1]}) not in SN")
                # check if substrate edge has enough bandwidth
                if result_graph.edges[s_edge[0], s_edge[1]]["weight"] < \
                        mapping_solution.VN.Graph.edges[v_edge['from'], v_edge['to']]["weight"]:
                    raise Exception(f"substrate edge ({s_edge[0]}, {s_edge[1]}) has bandwidth "
                                    f"{result_graph.edges[s_edge[0], s_edge[1]]['weight']}, but "
                                    f"virtual edge ({v_edge['from']}, {v_edge['to']}) requires "
                                    f"{mapping_solution.VN.Graph.edges[v_edge['from'], v_edge['to']]['weight']}"
                                    f" bandwidth")

                result_graph.edges[s_edge[0], s_edge[1]]["weight"] -= \
                    mapping_solution.VN.Graph.edges[v_edge['from'], v_edge['to']][
                        "weight"]

        return Network(result_graph, self.node_bound, self.cpu_bound, self.edge_bound, self.bandwidth_bound,
                       self.network_type)


class BINetwork(Network):
    def __init__(self, Graph, node_bound, cpu_bound, edge_bound, bandwidth_bound, network_type,
                 bi_type, beta, step):
        Network.__init__(self, Graph, node_bound, cpu_bound, edge_bound, bandwidth_bound, network_type)
        self.bi_type = bi_type
        self.beta = beta
        self.step = step
        self.min_cpu = 0.0
        self.max_cpu = 0.0
        self.mid_cpu = 0.0
        self.avg_cpu = 0.0
        self.node_num = 0
        self.max_neighbor_band = {}
        self.min_neighbor_band = {}
        self.mid_neighbor_band = {}
        self.avg_neighbor_band = {}
        self.deg = {}

        self.calculate_features()

    def plot(self):
        # print(f"{self.bi_type} BI, beta={self.beta}")
        plt.title(f"{self.bi_type} BI, beta={self.beta}")
        plot_graph_raw(self.Graph)

    def list_node_cpus(self):
        cpus = []
        for node in self.Graph.nodes:
            cpus.append(self.Graph.nodes[node]["cpu"])
        return cpus

    def list_node_neighbor_link_bandwidths(self, x):
        bands = []
        for neighbor, edge in self.Graph[x].items():
            bands.append(edge['weight'])
        return bands

    def calculate_features(self):
        # min_cpu, max_cpu, mid_cpu, avg_cpu, node_num, sum of bandwidth to neighbors
        cpus = np.array(self.list_node_cpus())
        self.min_cpu = np.min(cpus) if cpus.size > 0 else 0.0
        self.max_cpu = np.max(cpus) if cpus.size > 0 else 0.0
        self.mid_cpu = np.median(cpus) if cpus.size > 0 else 0.0
        self.avg_cpu = np.average(cpus) if cpus.size > 0 else 0.0
        self.node_num = self.Graph.number_of_nodes() if cpus.size > 0 else 0

        for node in self.Graph.nodes:
            bands = np.array(self.list_node_neighbor_link_bandwidths(node))
            self.max_neighbor_band[node] = np.max(bands) if bands.size > 0 else 0.0
            self.min_neighbor_band[node] = np.min(bands) if bands.size > 0 else 0.0
            self.mid_neighbor_band[node] = np.median(bands) if bands.size > 0 else 0.0
            self.avg_neighbor_band[node] = np.average(bands) if bands.size > 0 else 0.0
            self.deg[node] = self.Graph.degree[node] if bands.size > 0 else 0

    def output_features(self, s_node):
        if (self.Graph.number_of_nodes() > 0) and (
                not (s_node in self.max_neighbor_band and s_node in self.min_neighbor_band and
                     s_node in self.mid_neighbor_band and s_node in self.avg_neighbor_band and s_node in self.deg)):
            raise Exception(f"substrate node {s_node} not in bi {list(self.Graph.nodes)}")

        features = [self.beta, self.min_cpu, self.max_cpu, self.mid_cpu, self.avg_cpu, self.node_num,
                    self.max_neighbor_band.get(s_node, 0), self.min_neighbor_band.get(s_node, 0),
                    self.mid_neighbor_band.get(s_node, 0), self.avg_neighbor_band.get(s_node, 0),
                    self.deg.get(s_node, 0)]

        return features


class MappingSolution:
    def __init__(self, VN, node_map, virtual_links, mapped_links):
        self.VN = VN
        self.node_map = node_map
        self.virtual_links = virtual_links
        self.mapped_links = mapped_links
        self.v_node_revenue = {}
        self.v_node_cost = {}

        self.calculate_revenue_per_node(1.0)
        self.calculate_cost_per_node()

    def calculate_revenue_per_node(self, beta):
        for v_node in self.VN.Graph.nodes:
            self.v_node_revenue[v_node] = self.VN.Graph.nodes[v_node]["cpu"]
        for v_link in self.virtual_links:
            self.v_node_revenue[v_link["from"]] += beta * v_link["weight"]
            self.v_node_revenue[v_link["to"]] += beta * v_link["weight"]

    def calculate_cost_per_node(self):
        for v_node in self.VN.Graph.nodes:
            self.v_node_cost[v_node] = self.VN.Graph.nodes[v_node]["cpu"]
        for index in range(len(self.virtual_links)):
            v_link = self.virtual_links[index]
            s_path = self.mapped_links[index]
            self.v_node_cost[v_link["from"]] += len(s_path) * v_link["weight"]
            self.v_node_cost[v_link["to"]] += len(s_path) * v_link["weight"]

    def calculate_revenue(self, beta):
        cpu_revenue = 0
        band_revenue = 0

        for node in self.VN.Graph.nodes:
            cpu_revenue += self.VN.Graph.nodes[node]["cpu"]

        for u, v, weight in self.VN.Graph.edges.data("weight"):
            band_revenue += weight

        return cpu_revenue + beta * band_revenue

    def calculate_cost(self):
        cpu_cost = 0
        band_cost = 0

        for node in self.VN.Graph.nodes:
            cpu_cost += self.VN.Graph.nodes[node]["cpu"]

        for index in range(len(self.virtual_links)):
            band_cost += self.virtual_links[index]["weight"] * len(self.mapped_links[index])
            #print(len(self.mapped_links[index]))

        return cpu_cost + band_cost

    def calculate_revenue_cost_ratio(self, beta):
        return self.calculate_revenue(beta) / self.calculate_cost()


# TODO: implement me
def construct_network_from_std_in():
    pass


# construct_physical_network() constructs a multi-graph that denotes the substrate network
# randomly. No edges with same src and dest will be generated.
#
# attribute:
#   node_bound(int): max number of nodes in the graph
#   cpu_bound(int):  max number of cpus per node
#   edge_bound(int): max number of edges in the graph
#   bandwidth_bound(int): max bandwidth(MB) per edge
#
# returns:
#   G(nx.MultiGraph): a multi-graph denoting the substrate network
def construct_network_random(node_bound, cpu_bound, edge_bound, bandwidth_bound, network_type):
    G = nx.Graph()

    # generate available cpus for each node.
    # the multi-graph should contain at least one node.
    node_num = np.random.randint(3, node_bound + 1)
    # each node should have non-negative number of cpus.
    G.add_nodes_from([
        (i, {"cpu": np.random.randint(1, cpu_bound + 1)}) for i in range(0, node_num)
    ])

    # generate edges with available bandwidth.
    # the multi-graph should contain non-negative number of edges
    edge_num = np.random.randint(0, min(edge_bound, node_num * (node_num - 1) / 2) + 1)

    enable_multi_edge = True if (network_type == NetworkType.Virtual) else False
    edge_list = generate_edges(node_num, edge_num, bandwidth_bound, enable_multi_edge)
    G.add_weighted_edges_from(edge_list)

    return Network(G, node_bound, cpu_bound, edge_bound, bandwidth_bound, network_type)


def construct_network_input():
    G = nx.Graph()

    fact = 6
    node_num = 6
    G.add_nodes_from([
        (0, {"cpu": 10 * fact}),
        (1, {"cpu": 30 * fact}),
        (2, {"cpu": 40 * fact}),
        (3, {"cpu": 10 * fact}),
        (4, {"cpu": 20 * fact}),
        (5, {"cpu": 60 * fact}),
    ])

    G.add_weighted_edges_from([
        (0, 1, 15 * fact),
        (0, 2, 40 * fact),
        (0, 3, 25 * fact),
        (0, 4, 32 * fact),
        (1, 2, 17 * fact),
        (1, 4, 10 * fact),
        (2, 3, 32 * fact),
        (2, 4, 18 * fact),
        (2, 5, 14 * fact),
        (3, 4, 25 * fact),
        (3, 5, 16 * fact),
    ])

    return Network(G, node_bound=6, cpu_bound=60, edge_bound=11, bandwidth_bound=40, network_type=NetworkType.Substrate)


def generate_edges(node_num, edge_num, bandwidth_bound, enable_multi_edge):
    curr_edge_num = 0
    curr_edges = set()

    edge_list = []

    while curr_edge_num < edge_num:
        src = np.random.randint(0, node_num)
        dst = np.random.randint(0, node_num)
        while (node_num != 1) and (dst == src):
            dst = np.random.randint(0, node_num)

        if (not enable_multi_edge) and ((src, dst) in curr_edges):
            continue

        band = np.random.randint(1, bandwidth_bound + 1)
        edge_list.append((src, dst, band))
        curr_edges.add((src, dst))
        curr_edges.add((dst, src))

        curr_edge_num += 1
    return edge_list


def plot_graph_raw(G):
    pos = nx.spring_layout(G)

    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G, "cpu")
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()
