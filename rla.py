import functools
import time
from datetime import datetime

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

import copy

learning_rate = 0.05

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        # nn modules
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=4,
                out_channels=1,
                kernel_size=1),
            torch.nn.Softmax(dim=-1)
        )


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
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(state, SN=None, req=None, v_node=None, mapped_nodes=None):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    # state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state)).detach().numpy().tolist()[0][0]
    # print("state = ", state)


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

        policy.policy_history.append(0)
        return None, (SN.Graph.nodes[max_prob_s_node]["cpu"] - req.Graph.nodes[v_node]["cpu"]) / req.Graph.nodes[v_node]["cpu"]

    # print("available_s_node_probs = ", available_s_node_probs)
    cc = Categorical(torch.from_numpy(np.array([[available_s_node_probs]])))
    index = cc.sample()
    action = available_s_nodes[index.item()]
    #print(index, action)

    # Add log probability of our chosen action to our history
    if len(policy.policy_history) != 0:
        policy.policy_history.append(cc.log_prob(index).item())
    else:
        policy.policy_history = [cc.log_prob(index).item()]
    return action, 0


def update_policy():
    rewards = []

    for index in range(len(policy.reward_episode)):
        r = policy.reward_episode[index]

        revenue = policy.revenue_episode[index]
        cost = policy.cost_episode[index]
        size = policy.size_episode[index]
        solution = policy.solution_episode[index]

        if not policy.map_state_episode[index]:
            for _ in range(size):
                rewards.append(0)
        else:
            for v_node in solution.VN.Graph.nodes:
                rewards.append(revenue / cost)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    policy.policy_history = torch.from_numpy(np.array(policy.policy_history))
    #print(policy.policy_history, Variable(rewards))
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), dim=-1))
    loss_val = loss.item()
    #print("rla loss_val = ", loss_val)

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


def get_input_feature(SN, VN, v_node, used_s_node):
    input_feature = []
    feature_arr = []

    for s_node in SN.Graph.nodes:
        node_features = []

        node_features.append(SN.Graph.nodes[s_node]["cpu"])
        node_features.append(SN.Graph.degree[s_node])

        band_sum = 0
        for neighbor, edge in SN.Graph[s_node].items():
            band_sum += edge["weight"]
        node_features.append(band_sum)

        host_dists = []
        dist, path = bi.bfs_optimize_minimum(SN.Graph, source=s_node)
        for host_node in used_s_node:
            host_dists.append(dist[host_node])
        node_features.append(sum(host_dists) / (len(host_dists) + 1))

        feature_arr.append(node_features)

    input_feature.append(feature_arr)
    #print(input_feature)
    # print(input_feature)
    return torch.from_numpy(np.array(input_feature)).type(torch.FloatTensor).permute(0, 2, 1)


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
            #"lcf_bi": calculate_lcf(node_band_bis, node_mapping[u], node_mapping[v]),
            #"largest_beta_common_band_bi": calculate_largest_beta_common_bi(node_band_bis, node_mapping[u],
            #                                                                node_mapping[v])
        })

    # sort the virtual links
    # virtual_links.sort(key=functools.cmp_to_key(virtual_link_cmp))

    mapping_solution = []

    # map substrate path for each link
    for link in virtual_links:
        #common_bi = link["largest_beta_common_band_bi"]
        dist, paths = bi.bfs_optimize_minimum(SN.Graph, source=link["s_node_from"], target=link["s_node_to"],
                                              lowerbound=link["weight"],
                                              weight="weight")
        if link["s_node_to"] not in dist:
            # path not found
            return None, None, 0

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


def train_rla_converge(SN_raw, training_set, eps):
    batch_size = 10
    previous = 0
    start_time = time.time()
    epoch = 0

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
            # tmp_policy_history = policy.policy_history
            for node in req.Graph.nodes:
                # TODO: get input feature
                state = get_input_feature(SN, req, node, used_nodes)
                # print(type(state), state.shape, state)
                # TODO: forward propagation
                action, reward = select_action(state, SN=SN, req=req, v_node=node, mapped_nodes=used_nodes)
                # print(type(action), type(action.item()), action.item())

                if action is None:
                    map_reward = 0
                    # print("map_reward = ", map_reward)
                    node_map_success = False
                    continue

                SN.Graph.nodes[action]["cpu"] -= req.Graph.nodes[node]["cpu"]
                node_mapping[node] = action
                used_nodes.add(action)

            # step 3.2.2: link mapping
            virtual_links = []
            mapped_links = []
            if node_map_success:
                virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis,
                                                                          node_mapping)
                if virtual_links is None:
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

                policy.reward_episode.append(0)
                policy.map_state_episode.append(False)
                policy.solution_episode.append(None)
                policy.cost_episode.append(0)
                policy.revenue_episode.append(0)
                # policy.policy_history = tmp_policy_history

                SN = tmp_SN
            elif not link_map_success:
                #print(f"epoch {epoch}: link mapping failed, reward = {map_reward}")

                policy.reward_episode.append(0)
                policy.map_state_episode.append(False)
                policy.solution_episode.append(None)
                policy.cost_episode.append(0)
                policy.revenue_episode.append(0)
                SN = tmp_SN
            else:
               # print(
               #     f"epoch {epoch}: success, revenue = {mapping_solution.calculate_revenue(1.0)}, cost = {mapping_solution.calculate_cost()}, ratio = {mapping_solution.calculate_revenue_cost_ratio(1.0)}")

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
                print(f"rla epoch {epoch}: loss_val = {loss_val}")
                if abs(loss_val) < eps:
                    end_time = time.time()
                    return epoch, end_time - start_time
                previous = loss_val


def rla_train(SN_raw, training_set):
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
            # req.plot()

            map_reward = 0
            node_map_success = True
            link_map_success = True

            # step 3.2.1: node mapping
            node_cpu_bis = []
            node_band_bis = []

            used_nodes = set()
            # tmp_policy_history = policy.policy_history
            for node in req.Graph.nodes:
                # TODO: get input feature
                state = get_input_feature(SN, req, node, used_nodes)
                # print(type(state), state.shape, state)
                # TODO: forward propagation
                action, reward = select_action(state, SN=SN, req=req, v_node=node, mapped_nodes=used_nodes)
                # print(type(action), type(action.item()), action.item())

                if action is None:
                    map_reward = 0
                    # print("map_reward = ", map_reward)
                    node_map_success = False
                    continue

                SN.Graph.nodes[action]["cpu"] -= req.Graph.nodes[node]["cpu"]
                node_mapping[node] = action
                used_nodes.add(action)

            # step 3.2.2: link mapping
            virtual_links = []
            mapped_links = []
            if node_map_success:
                virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis,
                                                                          node_mapping)
                if virtual_links is None:
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
                print(f"epoch {epoch}: node mapping failed, reward = {map_reward}")

                policy.reward_episode.append(0)
                policy.map_state_episode.append(False)
                policy.solution_episode.append(None)
                policy.cost_episode.append(0)
                policy.revenue_episode.append(0)
                # policy.policy_history = tmp_policy_history

                SN = tmp_SN
            elif not link_map_success:
                print(f"epoch {epoch}: link mapping failed, reward = {map_reward}")

                policy.reward_episode.append(0)
                policy.map_state_episode.append(False)
                policy.solution_episode.append(None)
                policy.cost_episode.append(0)
                policy.revenue_episode.append(0)
                SN = tmp_SN
            else:
                print(
                    f"epoch {epoch}: success, revenue = {mapping_solution.calculate_revenue(1.0)}, cost = {mapping_solution.calculate_cost()}, ratio = {mapping_solution.calculate_revenue_cost_ratio(1.0)}")

                policy.reward_episode.append(0)
                policy.cost_episode.append(mapping_solution.calculate_cost())
                policy.revenue_episode.append(mapping_solution.calculate_revenue(1.0))
                policy.map_state_episode.append(True)
                policy.solution_episode.append(mapping_solution)
                # SN = tmp_Network

            # step 3.3: back propagation
            if index % batch_size == batch_size - 1:
                # print(f"request index = {index}")
                update_policy()


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


def rla(SN_raw, testing_set):
    SN = copy.deepcopy(SN_raw)
    rla_ratio = []
    rla_revenue = []
    rla_cost = []
    rla_succ = 0

    for req in testing_set:
        tmp_SN = copy.deepcopy(SN)
        node_mapping = {}
        node_map_success = True
        link_map_success = True

        # step 3.2.1: node mapping
        node_cpu_bis = []
        node_band_bis = []

        used_nodes = set()
        for node in req.Graph.nodes:
            state = get_input_feature(SN, req, node, used_nodes)
            action, reward = select_action_max(state, SN=SN, req=req, v_node=node, mapped_nodes=used_nodes)
            if action is None:
                rla_ratio.append(-1)
                node_map_success = False
                break
            SN.Graph.nodes[action]["cpu"] -= req.Graph.nodes[node]["cpu"]
            node_mapping[node] = action
            used_nodes.add(action)

        virtual_links = []
        mapped_links = []
        if node_map_success:
            virtual_links, mapped_links, lack_band_reward = map_links(SN, req, node_cpu_bis, node_band_bis, node_mapping)
            if virtual_links is None:
                rla_ratio.append(-1)
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
            rla_revenue.append(mapping_solution.calculate_revenue(1.0))
            rla_cost.append(mapping_solution.calculate_cost())
            rla_succ = rla_succ + 1

    return rla_ratio, rla_revenue, rla_cost, rla_succ
