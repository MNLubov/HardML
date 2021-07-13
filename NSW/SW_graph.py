#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


# In[2]:


def distance(pointA, documents): # -> np.ndarray:
    diff_matrix = documents - pointA
    
    if diff_matrix.ndim == 1:
        return np.linalg.norm(diff_matrix, axis = None)
    else:
        dist_euclidean = np.linalg.norm(diff_matrix, axis = 1)
        return dist_euclidean.reshape((-1,1))


# In[3]:


def create_sw_graph(
        data,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ):
    
    sw_graph = {}
    
    
    for i, point in enumerate(data):
        nodes = []
        
        #distance calc and sorting in descending and ascending order
        data_distance = dist_f(point, data)
        
        #check if number of nodes is small and therefore
        #zero distance (distance from document to itself) is excluded from consideration
        if len(data_distance) <= num_candidates_for_choice_long:
            num_candidates_for_choice_long = len(data_distance) - 1
        
        #sort in descending order
        candidates_long = np.argsort(data_distance, axis = 0)[::-1][:num_candidates_for_choice_long].flatten()
        idx_chosen_long = np.random.choice(candidates_long.shape[0], size=num_edges_long, replace = False)
        chosen_long = candidates_long[idx_chosen_long] #.tolist()
        #nodes.append(chosen_long)
        
        #sort in ascending order excluding zero element, 
        #to exclude zero distances, we consider array in the range [1:num_candidates_for_choice_short + 1]
        candidates_short = np.argsort(data_distance, axis = 0)[1:num_candidates_for_choice_short + 1].flatten()
        idx_chosen_short = np.random.choice(candidates_short.shape[0], size=num_edges_short, replace = False)
        chosen_short = candidates_short[idx_chosen_short] #.tolist()
        #nodes.extend(chosen_short)
        nodes_all = np.concatenate((chosen_long, chosen_short)).flatten()
        nodes.append(np.unique(nodes_all).tolist())
        
        unique_nodes = np.unique(nodes_all).tolist()
        sw_graph[i] = unique_nodes
        
    return sw_graph


# In[4]:


def nsw(query_point, all_documents, 
        graph_edges,
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance): #-> np.ndarray:
    
    k_nearest = []
    entry_nodes = []
    
    search_flag = False

    #cycle through all entry points
    i = 1
    while search_flag != True:
        
        entry_node = np.random.randint(len(graph_edges)) #np.random.choice(len(graph_edges), size = 1).item()
        #print('iteration = ',i, 'entry_node = ', entry_node)
        
        while entry_node in entry_nodes:
            entry_node = np.random.randint(len(graph_edges))
            #entry_node = np.random.choice(len(graph_edges), size = 1).item()     
        
        #setting values for the previous node and previos distance for local min search
        #entry node is the initial previous node
        previous_node_distance = dist_f(query_point, all_documents[entry_node])
        previous_node = entry_node
        #raise ValueError('Error1', all_documents.shape)
        
        entry_nodes.append(entry_node)
        
        min_flag = False
        
        while(min_flag != True):
            #formation of array of 'idx' node edges
            idx_edge_nodes = graph_edges[previous_node]
            edge_nodes = all_documents[idx_edge_nodes]
            
            #calculation of distance between query and edge nodes for current node
            dist_ = dist_f(query_point, edge_nodes)
            
            #document number/next node (with minimal distance to the query)
            idx_next_node = np.argsort(dist_, axis = 0)[0].flatten().item()
            next_node = idx_edge_nodes[idx_next_node]
            
            #minimal distance to the query
            query_node_distance = dist_[idx_next_node]

            if query_node_distance >= previous_node_distance:
                query_nearest_node = previous_node
                query_nearest_distance = previous_node_distance
                #adding nearest node if itn't already in k nearest neighbors
                if query_nearest_node not in k_nearest:
                    k_nearest.append(query_nearest_node)
                
                #stop searching local minima even node is already in k_nearest        
                min_flag = True
            else:
                previous_node = next_node
                previous_node_distance = query_node_distance
                #print()

        idx_next_nearest_nodes = graph_edges[query_nearest_node]
        next_nearest_nodes = all_documents[idx_next_nearest_nodes]
        dist_next_nearest = dist_f(query_point, next_nearest_nodes).flatten()
        
        for k in range(len(dist_next_nearest)):
            if dist_next_nearest[k] <= query_nearest_distance:
                if idx_next_nearest_nodes[k] not in k_nearest:
                    k_nearest.append(idx_next_nearest_nodes[k])
                
        if (len(k_nearest) >= search_k) and (i == num_start_points):
            search_flag = True
        if (len(k_nearest) >= search_k) and (i > num_start_points):
            search_flag = True
        
        i += 1

        
    k_nearest_arr = np.array(k_nearest) #[:search_k]
    dist_nearest_ = dist_f(query_point, all_documents[k_nearest_arr])
        
    #sort in descending order all nearest nodes, and return search_k nearest neighbors
    #idx_k_nearest = np.argsort(dist_nearest_, axis = 0)[::-1][:search_k].flatten()
    idx_k_nearest = np.argsort(dist_nearest_, axis = 0)[::-1].flatten()
              
    return k_nearest_arr


# In[ ]:




