import functools
import time
from datetime import datetime
import rla
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import networkx as nx
import random

import bi
import consts
import network
import math
import copy

env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

# Hyper-parameters
learning_rate = 0.05
gamma = 0.99

'''
input = torch.randn(32,35,256).permute(0,2,1)
print(type(input), input.shape)

state = torch.from_numpy(np.array(np.random.uniform(low=1, high=10, size=(4,)))).type(torch.FloatTensor)
print(type(state), state.shape, state)

exit(0)


arr0 = [1, 2, 3]
arr1 = [arr0]
arr2 = [arr1]

print(type(arr2), type(arr1), type(arr0))
input_arr = np.array(arr2)
print(type(input_arr), input_arr.shape, input_arr)
input_tensor = torch.from_numpy(input_arr).type(torch.FloatTensor)
print(type(input_tensor), input_tensor.shape, input_tensor)
'''

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        # nn modules
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=consts.CPUBIFeatureNum * consts.CPUStrides + consts.BandBIFeatureNum * consts.BandStrides + consts.NodeFeatureNum,
                out_channels=consts.BIConvKernelNum,
                kernel_size=consts.BIConvKernelSize),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=consts.BIConvKernelNum, out_channels=1,
                            kernel_size=consts.BIConvKernelSize),
            torch.nn.Softmax(dim=-1)
        )

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = []
        self.reward_episode = []
        self.map_state_episode = []
        self.cost_episode = []
        self.solution_episode = []
        self.revenue_episode = []
        self.size_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        return self.model(x)


policy = Policy()
print(policy.parameters())
for param in policy.parameters():
    print(param.shape)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action_max(state, SN=None, req=None, v_node=None, mapped_nodes=None):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    # state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state)).detach().numpy().tolist()[0][0]
    #print(v_node, state)

    max_prob = -1
    action = None
    for s_node in range(len(state)):
        if s_node in mapped_nodes or SN.Graph.nodes[s_node]["cpu"] < req.Graph.nodes[v_node]["cpu"] or (state[s_node] <= 0):
            continue
        if state[s_node] > max_prob:
            max_prob = state[s_node]
            action = s_node

    return action, 0


def select_action(state, SN=None, req=None, v_node=None, mapped_nodes=None):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    # state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state)).detach().numpy().tolist()[0][0]
    #print(v_node, state)


    available_s_node_probs = []
    available_s_nodes = []
    for s_node in range(len(state)):
        if s_node in mapped_nodes or SN.Graph.nodes[s_node]["cpu"] < req.Graph.nodes[v_node]["cpu"] or (state[s_node] <= 0):
            continue
        available_s_node_probs.append(state[s_node])
        available_s_nodes.append(s_node)

    if len(available_s_nodes) < 1:
        max_prob = state[0]
        max_prob_s_node = 0
        for s_node in range(len(state)):
            if state[s_node] > max_prob:
                max_prob = state[s_node]
                max_prob_s_node = s_node

        policy.policy_history.append(math.log(max_prob))
        return None, (SN.Graph.nodes[max_prob_s_node]["cpu"] - req.Graph.nodes[v_node]["cpu"]) / req.Graph.nodes[v_node]["cpu"]

    c = Categorical(torch.from_numpy(np.array([[available_s_node_probs]])))
    index = c.sample()
    action = available_s_nodes[index.item()]
    #print(index, action)

    # print(c.log_prob(index), c.log_prob(index).item())
    # Add log probability of our chosen action to our history
    if len(policy.policy_history) != 0:
        policy.policy_history.append(c.log_prob(index).item())
    else:
        policy.policy_history = [c.log_prob(index).item()]
    return action, 0


def update_policy():
    rewards = []

    # Discount future rewards back to the present using gamma
    sum_revenue = 0
    sum_cost = 0
    for index in range(len(policy.reward_episode)):
        r = policy.reward_episode[index]

        revenue = policy.revenue_episode[index]
        cost = policy.cost_episode[index]
        size = policy.size_episode[index]
        solution = policy.solution_episode[index]

        if not policy.map_state_episode[index]:
            for _ in range(size):
                rewards.append(r)
            sum_revenue = 0
            sum_cost = 0
        else:

            for v_node in solution.VN.Graph.nodes:
                rewards.append(solution.v_node_revenue[v_node] / solution.v_node_cost[v_node])
            '''
            sum_revenue += revenue
            sum_cost += cost

            #print(r, sum_revenue, sum_cost, revenue, cost)

            
            for _ in range(size):
                rewards.append(revenue / cost)
            '''



    '''
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)
    '''

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    policy.policy_history = torch.from_numpy(np.array(policy.policy_history))
    '''
    print("policy_history = ", policy.policy_history)
    print("reward = ", Variable(rewards))
    print("mul = ", torch.mul(policy.policy_history, Variable(rewards)))
    print("-mul = ", torch.mul(policy.policy_history, Variable(rewards)).mul(-1))
    print("sum = ", (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), dim=-1)))
    '''

    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), dim=-1))
    loss_val = loss.item()
    #print(loss, loss.item(), loss.data)
    #exit(0)

    # Update network weights
    # loss.backward(loss.clone().detach())
    loss.requires_grad = True
    loss.backward(loss.clone().detach())
    optimizer.step()
    optimizer.zero_grad()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = []
    policy.map_state_episode = []
    policy.reward_episode = []
    policy.cost_episode = []
    policy.solution_episode = []
    policy.revenue_episode = []
    policy.size_episode = []

    return loss_val

def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data[0])

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break


def construct_request_set(size):
    return [network.construct_network_random(4, 4, 5, 4, network.NetworkType.Virtual) for i in range(0, size)]


def get_input_feature(SN, VN, v_node):
    node_cpu_bis = dict()
    node_band_bis = dict()

    input_feature = []
    feature_arr = []

    for s_node in SN.Graph.nodes:
        node_cpu_bis[s_node] = []
        node_band_bis[s_node] = []
        node_features = []
        # 1. CPU stride BIs
        beta = 0
        delta = SN.cpu_bound / consts.CPUStrides
        for stride in range(0, consts.CPUStrides):
            # TODO:
            bi_network = SN.node_cpu_beta_mp[s_node][stride] #SN.get_bi_from_node_cpu(beta, s_node)
            node_features += bi_network.output_features(s_node)
            node_cpu_bis[s_node].append(bi_network)
            beta += delta

        # 2. cpu features
        node_features.append(SN.Graph.nodes[s_node]["cpu"])
        node_features.append(VN.Graph.nodes[v_node]["cpu"] - SN.Graph.nodes[s_node]["cpu"])

        # 2. Bandwidth stride BIs
        beta = 0
        delta = SN.bandwidth_bound / consts.BandStrides
        for stride in range(0, consts.BandStrides):
            # TODO:
            bi_network = SN.node_band_beta_mp[s_node][stride] #SN.get_bi_from_edge_band(beta, s_node)
            node_features += bi_network.output_features(s_node)
            node_band_bis[s_node].append(bi_network)
            beta += delta

        feature_arr.append(node_features)

    input_feature.append(feature_arr)
    # print(input_feature)
    return torch.from_numpy(np.array(input_feature)).type(torch.FloatTensor).permute(0, 2,
                                                                                     1), node_cpu_bis, node_band_bis


def virtual_link_cmp(left, right):
    left_bi = left["lcf_bi"]
    right_bi = right["lcf_bi"]

    if left_bi.beta != right_bi.beta:
        return left_bi.beta - right_bi.beta

    if left_bi.node_num != right_bi.node_num:
        return left_bi.node_num - right_bi.node_num

    if left["weight"] != right["weight"]:
        return right["weight"] - left["weight"]

    rand = random.randint(0, 1)
    if rand == 0:
        return -1
    else:
        return 1


def calculate_lcf(node_band_bis, s_node_u, s_node_v):
    for bi in node_band_bis[s_node_u]:
        if bi.Graph.has_node(s_node_u) and bi.Graph.has_node(s_node_v):
            return bi

    return None


def calculate_largest_beta_common_bi(node_band_bis, s_node_u, s_node_v):
    common_bi = None
    for bi in node_band_bis[s_node_u][::-1]:
        if bi.Graph.has_node(s_node_u) and bi.Graph.has_node(s_node_v):
            return bi


def map_links(SN, VN, node_cpu_bis, node_band_bis, node_mapping):
    # get all virtual links
    virtual_links = []
    virtual_links_sn_lcf_beta = dict()
    for u, v, weight in VN.Graph.edges.data('weight'):
        virtual_links.append({
            "from": u,
            "to": v,
            "s_node_from": node_mapping[u],
            "s_node_to": node_mapping[v],
            "weight": weight,
            "lcf_bi": calculate_lcf(node_band_bis, node_mapping[u], node_mapping[v]),
            "largest_beta_common_band_bi": calculate_largest_beta_common_bi(node_band_bis, node_mapping[u],
                                                                            node_mapping[v])
        })

    # sort the virtual links
    virtual_links.sort(key=functools.cmp_to_key(virtual_link_cmp))

    mapping_solution = []

    # map substrate path for each link
    for link in virtual_links:
        common_bi = link["largest_beta_common_band_bi"]
        dist, paths = bi.bfs_optimize_minimum(common_bi.Graph, source=link["s_node_from"], target=link["s_node_to"],
                                              lowerbound=link["weight"],
                                              weight="weight")
        if link["s_node_to"] not in dist:
            # path not found
            #print(f"cannot map link ({link['from']}, {link['to']}, {link['weight']})")
            maximal_min = bi.find_maximal_path_minimal_edge(common_bi.Graph, source=link["s_node_from"],
                                                            target=link["s_node_to"], upperbound=link["weight"],
                                                            weight="weight")
            return None, None, (maximal_min - link["weight"]) / link["weight"]

        passed_substrate_nodes = paths[link["s_node_to"]]
        current_substrate_node = link["s_node_from"]
        substrate_path = []
        for s_node in passed_substrate_nodes:
           # print(f"DEBUG current_substrate_node = {current_substrate_node}, s_node = {s_node}. "
           #       f"link_required_band = {link['weight']}")
            SN.Graph.edges[current_substrate_node, s_node]["weight"] -= link["weight"]

            substrate_path.append((current_substrate_node, s_node))
            current_substrate_node = s_node

        mapping_solution.append(substrate_path)

    return virtual_links, mapping_solution, 0


# step 1: generate substrate graph
SN_raw = network.construct_network_input()
# SN_raw.plot()
'''
g2 = nx.subgraph(SN_raw.Graph, nbunch=None)
g3 = nx.subgraph(SN_raw.Graph, nbunch=set())
print(list(g2.nodes))
print(list(g3.edges), "edges")
print(list(g3.nodes), "nodes")
exit(0)
'''

'''
for node in SN_raw.Graph.nodes:
    print(node, SN_raw.Graph.nodes[node]["cpu"])
'''


Ns = SN_raw.get_all_bis()
for N in Ns:
    pass
    # N.plot()

def construct_converge_request_set(size):
    return [network.construct_network_random(3, 2, 4, 1, network.NetworkType.Virtual) for i in range(0, size)]

# step 2: generate training set and testing set
training_set_size = 50
testing_set_size = 80
training_set = construct_request_set(training_set_size)
testing_set = construct_request_set(testing_set_size)

testing_sets = [construct_request_set(size) for size in range(30, 101, 10)]
converge_testing_sets = [construct_converge_request_set(size) for size in range(30, 101, 10)]


def train_VNERL_converge(training_set, eps):
    start_time = time.time()

    epoch = 0
    previous = 0

    batch_size = 10

    while True:
        epoch += 1

        # step 3.1: initialization
        SN = copy.deepcopy(SN_raw)

        # step 3.2: batch training
        for index in range(len(training_set)):
            tmp_SN = copy.deepcopy(SN)

            node_mapping = {}

            req = training_set[index]
            # req.plot()

            map_reward = 0
            node_map_success = True
            link_map_success = True

            # step 3.2.1: node mapping
            node_cpu_bis = []
            node_band_bis = []

            used_nodes = set()
            SN.calculate_all_nodes_beta_bis()
            for node in req.Graph.nodes:
                # TODO: get input feature
                state, node_cpu_bis, node_band_bis = get_input_feature(SN, req, node)
                # print(type(state), state.shape, state)
                # TODO: forward propagation
                action, reward = select_action(state, SN=SN, req=req, v_node=node, mapped_nodes=used_nodes)
                # print(type(action), type(action.item()), action.item())

                if action is None:
                    map_reward = min(map_reward, reward)
                    # print("map_reward = ", map_reward)
                    node_map_success = False
                    continue

                SN.Graph.nodes[action]["cpu"] -= req.Graph.nodes[node]["cpu"]
                node_mapping[node] = action
                used_nodes.add(action)
                SN.calculate_all_nodes_beta_bis()

            # step 3.2.2: link mapping
            virtual_links = []
            mapped_links = []
            if node_map_success:
                virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis,
                                                                          node_mapping)
                if lack_band_reward != 0:
                    map_reward = lack_band_reward
                    link_map_success = False

            # step 3.2.3: map node and link
            tmp_Network = None
            mapping_solution = None
            if node_map_success and link_map_success:
                mapping_solution = network.MappingSolution(req, node_mapping, virtual_links, mapped_links)
                tmp_Network = tmp_SN.after(mapping_solution)

            # step 3.2.4: step
            policy.size_episode.append(len(req.Graph.nodes))
            if not node_map_success:
                #print(f"epoch {epoch}: node mapping failed, reward = {map_reward}")

                policy.reward_episode.append(map_reward)
                policy.map_state_episode.append(False)
                policy.solution_episode.append(None)
                policy.cost_episode.append(0)
                policy.revenue_episode.append(0)
                # policy.policy_history = tmp_policy_history

                SN = tmp_SN
            elif not link_map_success:
                #print(f"epoch {epoch}: link mapping failed, reward = {map_reward}")

                policy.reward_episode.append(map_reward)
                policy.map_state_episode.append(False)
                policy.solution_episode.append(None)
                policy.cost_episode.append(0)
                policy.revenue_episode.append(0)
                SN = tmp_SN
            else:
                # print(f"epoch {epoch}: success, revenue = {mapping_solution.calculate_revenue(1.0)}, cost = {mapping_solution.calculate_cost()}, ratio = {mapping_solution.calculate_revenue_cost_ratio(1.0)}")

                policy.reward_episode.append(0)
                policy.cost_episode.append(mapping_solution.calculate_cost())
                policy.revenue_episode.append(mapping_solution.calculate_revenue(1.0))
                policy.map_state_episode.append(True)
                policy.solution_episode.append(mapping_solution)
                # SN = tmp_Network

            # step 3.3: back propagation
            if index % batch_size == batch_size - 1:
                # print(f"request index = {index}")
                loss_val = update_policy()
                print(f"ice epoch {epoch}: loss_val = {loss_val}")
                if abs(loss_val) < eps:
                    end_time = time.time()
                    return epoch, end_time - start_time
                previous = loss_val


if True:
    x_data = [len(training_set) for training_set in converge_testing_sets]

    eps = 0.001
    ice_epochs = []
    ice_times = []
    rla_epochs = []
    rla_times = []
    for ts in converge_testing_sets:
        ice_epoch, ice_time = train_VNERL_converge(ts, eps)
        rla_epoch, rla_time = rla.train_rla_converge(copy.deepcopy(SN_raw), ts, eps)
        print(f"size = {len(ts)}, ice_epoch = {ice_epoch}, ice_time = {ice_time}")
        print(f"size = {len(ts)}, rla_epoch = {rla_epoch}, rla_time = {rla_time}")
        ice_epochs.append(ice_epoch)
        ice_times.append(ice_time)
        rla_epochs.append(rla_epoch)
        rla_times.append(rla_time)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300

    # VNERL_epoch_line = plt.plot(x_data, ice_epochs, color='blue', linewidth=2.0, linestyle='-', marker='v', label='ICE')
    # rla_epoch_line = plt.plot(x_data, rla_epochs, color='green', linewidth=2.0, linestyle='-', marker='^', label='RLA')
    # plt.xlabel('Training Set Size (Number of Virtual Network Requests)')
    # plt.ylabel('Number of Training Epochs')
    # plt.legend()
    # # plt.show()
    # plt.savefig('epochs.png', dpi=300)

    # plt.cla()

    # VNERL_ct_line = plt.plot(x_data, ice_times, color='blue', linewidth=2.0, linestyle='-', marker='v',
    #                             label='ICE')
    # rla_ct_line = plt.plot(x_data, rla_times, color='green', linewidth=2.0, linestyle='-', marker='^',
    #                           label='RLA')
    # plt.xlabel('Training Set Size (Number of Virtual Network Requests)')
    # plt.ylabel('Convergence Time')
    # plt.legend()
    # # plt.show()
    # plt.savefig('ct.png', dpi=300)

    # plt.cla()
    # exit(0)

rla.rla_train(copy.deepcopy(SN_raw), training_set)

# step 3: training
num_epoch = 150
batch_size = 3
for epoch in range(0, num_epoch):
    # step 3.1: initialization
    SN = copy.deepcopy(SN_raw)

    # step 3.2: batch training
    for index in range(len(training_set)):
        tmp_SN = copy.deepcopy(SN)

        node_mapping = {}

        req = training_set[index]
        #req.plot()

        map_reward = 0
        node_map_success = True
        link_map_success = True

        # step 3.2.1: node mapping
        node_cpu_bis = []
        node_band_bis = []

        used_nodes = set()
        SN.calculate_all_nodes_beta_bis()
        for node in req.Graph.nodes:
            # TODO: get input feature
            state, node_cpu_bis, node_band_bis = get_input_feature(SN, req, node)
            # print(type(state), state.shape, state)
            # TODO: forward propagation
            action, reward = select_action(state, SN=SN, req=req, v_node=node, mapped_nodes=used_nodes)
            # print(type(action), type(action.item()), action.item())

            if action is None:
                map_reward = min(map_reward, reward)
                #print("map_reward = ", map_reward)
                node_map_success = False
                continue

            SN.Graph.nodes[action]["cpu"] -= req.Graph.nodes[node]["cpu"]
            node_mapping[node] = action
            used_nodes.add(action)
            SN.calculate_all_nodes_beta_bis()

        # step 3.2.2: link mapping
        virtual_links = []
        mapped_links = []
        if node_map_success:
            virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis, node_mapping)
            if lack_band_reward != 0:
                map_reward = lack_band_reward
                link_map_success = False

        # step 3.2.3: map node and link
        tmp_Network = None
        mapping_solution = None
        if node_map_success and link_map_success:
            mapping_solution = network.MappingSolution(req, node_mapping, virtual_links, mapped_links)
            tmp_Network = tmp_SN.after(mapping_solution)

        # step 3.2.4: step
        policy.size_episode.append(len(req.Graph.nodes))
        if not node_map_success:
            #print(f"epoch {epoch}: node mapping failed, reward = {map_reward}")

            policy.reward_episode.append(map_reward)
            policy.map_state_episode.append(False)
            policy.solution_episode.append(None)
            policy.cost_episode.append(0)
            policy.revenue_episode.append(0)
            #policy.policy_history = tmp_policy_history

            SN = tmp_SN
        elif not link_map_success:
            #print(f"epoch {epoch}: link mapping failed, reward = {map_reward}")

            policy.reward_episode.append(map_reward)
            policy.map_state_episode.append(False)
            policy.solution_episode.append(None)
            policy.cost_episode.append(0)
            policy.revenue_episode.append(0)
            SN = tmp_SN
        else:
            print(f"epoch {epoch}: success, revenue = {mapping_solution.calculate_revenue(1.0)}, cost = {mapping_solution.calculate_cost()}, ratio = {mapping_solution.calculate_revenue_cost_ratio(1.0)}")

            policy.reward_episode.append(0)
            policy.cost_episode.append(mapping_solution.calculate_cost())
            policy.revenue_episode.append(mapping_solution.calculate_revenue(1.0))
            policy.map_state_episode.append(True)
            policy.solution_episode.append(mapping_solution)
            # SN = tmp_Network

        # step 3.3: back propagation
        if index % batch_size == batch_size - 1:
            #print(f"request index = {index}")
            update_policy()


def VNERL(testing_set):
    SN = copy.deepcopy(SN_raw)
    VNERL_ratio = []
    VNERL_revenue = []
    VNERL_cost = []
    VNERL_succ = 0

    for req in testing_set:
        tmp_SN = copy.deepcopy(SN)
        node_mapping = {}
        node_map_success = True
        link_map_success = True

        # step 3.2.1: node mapping
        node_cpu_bis = []
        node_band_bis = []

        used_nodes = set()
        SN.calculate_all_nodes_beta_bis()
        for node in req.Graph.nodes:
            state, node_cpu_bis, node_band_bis = get_input_feature(SN, req, node)
            action, reward = select_action_max(state, SN=SN, req=req, v_node=node, mapped_nodes=used_nodes)
            if action is None:
                VNERL_ratio.append(-1)
                node_map_success = False
                break
            SN.Graph.nodes[action]["cpu"] -= req.Graph.nodes[node]["cpu"]
            node_mapping[node] = action
            used_nodes.add(action)
            SN.calculate_all_nodes_beta_bis()

        virtual_links = []
        mapped_links = []
        if node_map_success:
            virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis, node_mapping)
            if lack_band_reward != 0:
                VNERL_ratio.append(-1)
                map_reward = lack_band_reward
                link_map_success = False

        tmp_Network = None
        mapping_solution = None
        if node_map_success and link_map_success:
            mapping_solution = network.MappingSolution(req, node_mapping, virtual_links, mapped_links)
            tmp_Network = tmp_SN.after(mapping_solution)

        if not node_map_success:
            SN = tmp_SN
        elif not link_map_success:
            SN = tmp_SN
        else:
            VNERL_revenue.append(mapping_solution.calculate_revenue(1.0))
            VNERL_cost.append(mapping_solution.calculate_cost())
            VNERL_succ = VNERL_succ + 1

    return VNERL_ratio, VNERL_revenue, VNERL_cost, VNERL_succ


def HNO_Compare(a, b):
    if a["min-bi-nodes"] != b["min-bi-nodes"]:
        return a["min-bi-nodes"] - b["min-bi-nodes"]
    if a["cpu"] != b["cpu"]:
        return b["cpu"] - a["cpu"]
    rand = random.randint(0, 1)
    if rand == 0:
        return -1
    else:
        return 1


def calculate_min_bi_nodes(SN, beta):
    bi_nodes = []
    for node in SN.Graph.nodes:
        bi_nodes.append(len(bi.construct_single_bi_from_node_cpu(SN.Graph, beta, node)))

    return min(bi_nodes)


def HVNO(SN, VN):
    node_list = []
    for node in VN.Graph.nodes:
        node_list.append({
            "id": node,
            "cpu": VN.Graph.nodes[node]["cpu"],
            "min-bi-nodes": calculate_min_bi_nodes(SN, VN.Graph.nodes[node]["cpu"])
        })

    node_list.sort(key=functools.cmp_to_key(HNO_Compare))

    return node_list


def HNM(SN, VN, node_list):
    node_mapping = {}
    used_s_nodes = set()
    for node in node_list:
        min_s_node = -1
        min_splitting = 0
        max_cpu = 0

        s_nodes = []
        for s_node in SN.Graph.nodes:
            s_nodes.append(s_node)
        random.shuffle(s_nodes)

        for s_node in s_nodes:
            if SN.Graph.nodes[s_node]["cpu"] < VN.Graph.nodes[node["id"]]["cpu"] or (s_node in used_s_nodes):
                continue
            # calculate splitting
            before = len(bi.construct_bi_cpu(SN.Graph, VN.Graph.nodes[node["id"]]["cpu"]))
            SN.Graph.nodes[s_node]["cpu"] -= VN.Graph.nodes[node["id"]]["cpu"]
            after = len(bi.construct_bi_cpu(SN.Graph, VN.Graph.nodes[node["id"]]["cpu"]))
            SN.Graph.nodes[s_node]["cpu"] += VN.Graph.nodes[node["id"]]["cpu"]
            splitting = after - before

            cpu = SN.Graph.nodes[s_node]["cpu"]

            if min_s_node == -1 or splitting < min_splitting or (splitting == min_splitting and cpu > max_cpu):
                min_s_node = s_node
                min_splitting = splitting
                max_cpu = cpu

        node_mapping[node["id"]] = min_s_node
        if min_s_node == -1 or SN.Graph.nodes[min_s_node]["cpu"] < VN.Graph.nodes[node["id"]]["cpu"]:
            return None
        SN.Graph.nodes[min_s_node]["cpu"] -= VN.Graph.nodes[node["id"]]["cpu"]
        used_s_nodes.add(min_s_node)

    return node_mapping


request_revenue = {}
sum_bandwidth = {}


def DHRO_cmp(a, b):
    if request_revenue[a] != request_revenue[b]:
        return request_revenue[b] - request_revenue[a]
    if sum_bandwidth[a] != sum_bandwidth[b]:
        return sum_bandwidth[b] - sum_bandwidth[a]
    return -1


def presto(testing_set):
    presto_ratio = []
    presto_revenue = []
    presto_cost = []
    presto_succ = 0
    SN = copy.deepcopy(SN_raw)

    # 0. order testing set
    for req in testing_set:
        request_revenue[req] = req.sum_cpu_band()
        sum_bandwidth[req] = req.sum_band()

    testing_set.sort(key=functools.cmp_to_key(DHRO_cmp))

    for req in testing_set:
        node_map_success = True
        link_map_success = True

        tmp_SN = copy.deepcopy(SN)

        # 1. virtual node ordering
        node_list = HVNO(SN, req)

        # 2. virtual node mapping
        node_mapping = HNM(SN, req, node_list)
        if node_mapping is None:
            presto_ratio.append(-1)
            node_map_success = False

        # 3. virtual link mapping
        virtual_links = []
        mapped_links = []
        if node_map_success:
            SN.calculate_all_nodes_beta_bis()
            _, node_cpu_bis, node_band_bis = get_input_feature(SN, req, 0)

            virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis,
                                                                      node_mapping)
            if lack_band_reward != 0:
                presto_ratio.append(-1)
                link_map_success = False

        tmp_Network = None
        mapping_solution = None
        if node_map_success and link_map_success:
            mapping_solution = network.MappingSolution(req, node_mapping, virtual_links, mapped_links)
            tmp_Network = tmp_SN.after(mapping_solution)

        # 4. calculate reward
        if not node_map_success:
            SN = tmp_SN
        elif not link_map_success:
            SN = tmp_SN
        else:
            presto_revenue.append(mapping_solution.calculate_revenue(1.0))
            presto_cost.append(mapping_solution.calculate_cost())
            presto_succ += 1

    return presto_ratio, presto_revenue, presto_cost, presto_succ





def single_training(testing_sets):
    SN = copy.deepcopy(SN_raw)
    presto_succs = []
    presto_overall_ratio = []
    presto_time = []
    presto_costs = []
    presto_revenues = []
    VNERL_succs = []
    VNERL_overall_ratio = []
    VNERL_time = []
    VNERL_costs = []
    VNERL_revenues = []
    rla_succs = []
    rla_overall_ratio = []
    rla_time = []
    rla_costs = []
    rla_revenues = []
    x_data = [len(testing_set) for testing_set in testing_sets]
    for testing_set in testing_sets:
        presto_start = time.time()
        presto_ratio, presto_revenue, presto_cost, presto_succ = presto(testing_set)
        presto_end = time.time()
        presto_time.append(presto_end - presto_start)

        VNERL_start = time.time()
        VNERL_ratio, VNERL_revenue, VNERL_cost, VNERL_succ = VNERL(testing_set)
        VNERL_end = time.time()
        VNERL_time.append(VNERL_end - VNERL_start)


        rla_start = time.time()
        rla_ratio, rla_revenue, rla_cost, rla_succ = rla.rla(copy.deepcopy(SN_raw), testing_set)
        rla_end = time.time()
        rla_time.append(rla_end - rla_start)


        presto_succs.append(presto_succ / len(testing_set))
        VNERL_succs.append(VNERL_succ / len(testing_set))
        rla_succs.append(rla_succ / len(testing_set))
        presto_overall_ratio.append(sum(presto_revenue) / sum(presto_cost))
        VNERL_overall_ratio.append(sum(VNERL_revenue) / sum(VNERL_cost))

        if sum(rla_cost) == 0:
            rla_overall_ratio.append(0)
        else:
            rla_overall_ratio.append(sum(rla_revenue) / sum(rla_cost))

        presto_costs.append(sum(presto_cost) / presto_succ)
        presto_revenues.append(sum(presto_revenue))

        VNERL_costs.append(sum(VNERL_cost) / VNERL_succ)
        VNERL_revenues.append(sum(VNERL_revenue))

        if rla_succ != 0:
            rla_costs.append(sum(rla_cost) / rla_succ)
        else:
            rla_costs.append(0)
        rla_revenues.append(sum(rla_revenue))


        print(f"testing set size = {len(testing_set)}:")
        print(f"presto_sr = {presto_succ / len(testing_set)}, presto_ratio = {sum(presto_revenue) / sum(presto_cost)}, avg_presto_cost={sum(presto_cost) / presto_succ}, presto_revenue={sum(presto_revenue)}")
        print(f"VNERL_sr = {VNERL_succ / len(testing_set)}, VNERL_ratio = {sum(VNERL_revenue) / sum(VNERL_cost)}, avg_VNERL_cost={sum(VNERL_cost) / VNERL_succ}, VNERL_revenue={sum(VNERL_revenue)}")

    '''
        print(
            f"rla_sr = {rla_succ / len(testing_set)}, rla_ratio = {sum(rla_revenue) / sum(rla_cost)}, avg_rla_cost={sum(rla_cost) / rla_succ}, rla_revenue={sum(rla_revenue)}")
    '''

    print("sr: ", rla_succs)
    print("rvn: ", rla_revenues)
    print("ebdc: ", rla_costs)
    print("ratio: ", rla_overall_ratio)

    return np.average(presto_succs), np.average(presto_overall_ratio), np.average(presto_revenues), np.average(presto_costs), np.average(presto_time),\
           np.average(VNERL_succs), np.average(VNERL_overall_ratio), np.average(VNERL_revenues), np.average(VNERL_costs), np.average(VNERL_time),\
           np.average(rla_succs), np.average(rla_overall_ratio), np.average(rla_revenues), np.average(rla_costs), np.average(rla_time),



presto_succs = []
presto_overall_ratio = []
presto_time = []
presto_costs = []
presto_revenues = []
VNERL_succs = []
VNERL_overall_ratio = []
VNERL_time = []
VNERL_costs = []
VNERL_revenues = []
rla_succs = []
rla_overall_ratio = []
rla_time = []
rla_costs = []
rla_revenues = []
repeat_num = 10
x_data = [len(testing_set) for testing_set in testing_sets]
for size in x_data:
    testing_sets = [construct_request_set(size) for i in range(repeat_num)]
    ps, por, pr, pc, pt, \
    vs, vor, vr, vc, vt, \
    rs, ror, rr, rc, rt = single_training(testing_sets)
    presto_succs.append(ps)
    presto_overall_ratio.append(por)
    presto_revenues.append(pr)
    presto_costs.append(pc)
    presto_time.append(pt)

    VNERL_succs.append(vs)
    VNERL_overall_ratio.append(vor)
    VNERL_revenues.append(vr)
    VNERL_costs.append(vc)
    VNERL_time.append(vt)

    rla_succs.append(rs)
    rla_overall_ratio.append(ror)
    rla_revenues.append(rr)
    rla_costs.append(rc)
    rla_time.append(rt)


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

presto_succ_line = plt.plot(x_data, presto_succs, color='red', linewidth=2.0, linestyle='-', marker='o', label='Presto')
VNERL_succ_line = plt.plot(x_data, VNERL_succs, color='blue', linewidth=2.0, linestyle='-', marker='v', label='ICE')
rla_succ_line = plt.plot(x_data, rla_succs, color='green', linewidth=2.0, linestyle='-', marker='^', label='RLA')
plt.xlabel('Number of Virtual Network Requests')
plt.ylabel('Acceptance Ratio')
plt.legend()
# plt.show()
plt.savefig('sr.png', dpi=300)

plt.cla()

presto_ratio_line = plt.plot(x_data, presto_overall_ratio, color='red', linewidth=2.0, linestyle='-', marker='o', label='Presto')
VNERL_ratio_line = plt.plot(x_data, VNERL_overall_ratio, color='blue', linewidth=2.0, linestyle='-', marker='v', label='ICE')
rla_ratio_line = plt.plot(x_data, rla_overall_ratio, color='green', linewidth=2.0, linestyle='-', marker='^', label='RLA')
plt.xlabel('Number of Virtual Network Requests')
plt.ylabel('Revenue-to-Cost Ratio')
plt.legend()
# plt.show()
plt.savefig('ratio.png', dpi=300)

plt.cla()

presto_compute_line = plt.plot(x_data, presto_time, color='red', linewidth=2.0, linestyle='-', marker='o', label='Presto')
VNERL_compute_line = plt.plot(x_data, VNERL_time, color='blue', linewidth=2.0, linestyle='-', marker='v', label='ICE')
rla_compute_line = plt.plot(x_data, rla_time, color='green', linewidth=2.0, linestyle='-', marker='^', label='RLA')
plt.xlabel('Number of Virtual Network Requests')
plt.ylabel('Time Spent Handling Requests')
plt.legend()
# plt.show()
plt.savefig('compute.png', dpi=300)

plt.cla()

presto_cost_line = plt.plot(x_data, presto_costs, color='red', linewidth=2.0, linestyle='-', marker='o', label='Presto')
VNERL_cost_line = plt.plot(x_data, VNERL_costs, color='blue', linewidth=2.0, linestyle='-', marker='v', label='ICE')
rla_cost_line = plt.plot(x_data, rla_costs, color='green', linewidth=2.0, linestyle='-', marker='^', label='RLA')
plt.xlabel('Number of Virtual Network Requests')
plt.ylabel('Average Embedding Cost')
plt.legend()
# plt.show()
plt.savefig('ebdc.png', dpi=300)

plt.cla()

presto_revenue_line = plt.plot(x_data, presto_revenues, color='red', linewidth=2.0, linestyle='-', marker='o', label='Presto')
VNERL_revenue_line = plt.plot(x_data, VNERL_revenues, color='blue', linewidth=2.0, linestyle='-', marker='v', label='ICE')
rla_revenue_line = plt.plot(x_data, rla_revenues, color='green', linewidth=2.0, linestyle='-', marker='^', label='RLA')
plt.xlabel('Number of Virtual Network Requests')
plt.ylabel('Revenue')
plt.legend()
# plt.show()
plt.savefig('rvn.png', dpi=300)

plt.cla()
