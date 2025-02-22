# # Test computation of minimal capacity

from reachability_examples import ultimate
from fimdp.energy_solvers import AS_REACH
from fimdp.mincap_solvers import bin_search
#from fimdpenv.MarsEnv import MarsEnv
import matplotlib.pyplot as plt
from networkx.algorithms import tournament
from networkx.algorithms import cycles
#import MarsEnv
import numpy as np
import scipy.optimize
from scipy.optimize import linear_sum_assignment
import networkx as nx
import math
from networkx.algorithms import bipartite
from networkx.algorithms import traversal
# In the following example, the minimal capacity is 15 for Buchi objective defined by the blue states.



#Compute the minimum cost of the assignments by bisection
def compute_min_cost_bisection(Agent_graph,num_agents):
    """
    :param Agent_graph: Networkx graph for the targets
    :type Agent_graph: Networkx undirected graph
    :param num_agents: Number of agents for assignments
    :return:
    path_list: Lists of paths for each agent
    cost_obtained: Maximum cost of the paths
    """

    #Generate minimum spanning tree
    tree=generate_minimumspanning_tree(Agent_graph)
    #Prints the ordered edges in the tree
    #print(sorted(tree.edges(data=True)))

    #Compute approximate Eulerian tour
    tour_min=min_hamilton(tree)
    #Prints the edges of eulerian tour
    #print("Edges of the tour",tour_min)
    #Compute the overall cost
    cost_tour = compute_cost(Agent_graph, tour_min)
    print("Cost of the min_tour:", cost_tour)

    #Assign parts of the min cost eulerian tour to each agent by bisection
    path_list,cost_obtained=bisection_loop(Agent_graph,tour_min,num_agents)

    #Print list of paths, number of paths, max_cost
    print("paths for each agent",path_list)
    print("Number of total paths",len(path_list))

    print("Max cost of paths", 4*cost_obtained)
    return(path_list,cost_obtained)


def bisection_loop(Agent_graph,tour_min,num_agents):
    """
    :param Agent_graph: Networkx graph for the targets
    :type Agent_graph: Networkx undirected graph
    :param tour_min: List of nodes for the approximate minimum cost Eulerian tour
    :type tour_min: List of nodes
    :param num_agents: Number of agents for assignments
    :return:
    all_agentlist: List of paths for each agent
    cost_bisec: Maximum cost of the paths of the agents
    """

    #Bisection over optimal costcost
    cost_low=0
    cost_upper=1e4
    best_cost=cost_upper
    returnflag=False
    while cost_upper-cost_low>1e-5 or not returnflag:
        #safeguard to ensure that the solution returns a valid assignment
        if abs(cost_upper - cost_low) <= 1e-5:
            cost_upper = best_cost
            cost_low = best_cost
            #print("")

        cost_bisec = (cost_upper + cost_low) / 2
        #Initialze lists for agents
        cost_agent=0
        all_agentlist=[]
        agent_list=[]
        total_list=[]
        #Go over the eulerian tour
        for i in range(len(tour_min)):

            #check if the initial target is covered by any agent
            if not tour_min[i] in total_list:

                # add the target to the agent
                agent_list.append(tour_min[i])
                # add the target to the all covered target lists
                total_list.append(tour_min[i])
                if i<len(tour_min)-1:
                    # update the cost if the next capacity is higher than previous one
                    if cost_agent < Agent_graph[tour_min[i]][tour_min[i + 1]]['weight']:
                        cost_agent = Agent_graph[tour_min[i]][tour_min[i + 1]]['weight']
                    if cost_agent < Agent_graph[tour_min[len(tour_min) - 1]][tour_min[0]]['weight']:
                        cost_agent = Agent_graph[tour_min[len(tour_min) - 1]][tour_min[0]]['weight']


            # If the cost exceeds 6x of the optimal cost, break the tour of this agent
            if cost_agent>6*((cost_upper+cost_low)/2):
                all_agentlist.append(agent_list)
                for item in agent_list:
                    total_list.append(item)
                agent_list=[]
                cost_agent=0
        #apennd tour
        if len(agent_list)>0:
            all_agentlist.append(agent_list)

        #If the number of tours is less or equal than the
        # number of agents, we increase our guess of the optimal cost
        #add flags to ensure that the solution is valid
        if len(all_agentlist)<=num_agents:
            returnflag=True
            best_cost= cost_bisec
            #print(best_cost)
        else:
            returnflag=False
        if len(all_agentlist)>num_agents:
            cost_low=cost_bisec

        #Else, we lower our guess of optimal cost
        else:
            cost_upper=cost_bisec

    #add empty assignments
    while len(all_agentlist)<num_agents:
        all_agentlist.append([])

    #Returns the paths for each agent, and the optimal cost,
    # which is guaranteed to be a 4 approximation for graphs that satisfies triangular inequality
    #print(tour_min)
    return (all_agentlist,cost_bisec)


#Computes an approximately optimal hamiltonian path from a minimum spanning tree
def min_hamilton(tree):
    """
    :param tree: List of edges in the minimum spanning tree
    :type: tree: Set of networkx edges
    :return:
    step_list: Ordered nodes of the Eulerian tour
    """

    step_list=[]
    # Go over the edges of tree, add to the Eulerian tour if they are not already present
    for item,item2,weight in (tree.edges(data=True)):
        if item not in step_list:
            step_list.append(item)

        if item2 not in step_list:
            step_list.append(item2)
    return(step_list)

def generate_Graph(T,init_state):
    """
    :param T: set of targets: which is a proxy here
    :param: set of initial states:
    :return:
    Agent_graph: Networkx graph with nodes representing each target, edges representing cost
    """
    #Generate nodes for each target
    Agent_graph= nx.DiGraph()

    count=0
    for item in T:
        count=count+1
        Agent_graph.add_node((item))

    agent_target_graph=nx.DiGraph()
    for item in T:
        count=count+1
        agent_target_graph.add_node((item))
    for item in init_state:
        count=count+1
        agent_target_graph.add_node((item))
    return Agent_graph,agent_target_graph


#Computes minimum spanning tree, used to generate minimum cost path
# def generate_minimumspanning_tree(Agent_graph):
#     """
#
#     :param Agent_graph: Networkx graph for the targets
#     :type Agent_graph: Networkx undirected graph
#     :return:
#     Tree: minimum spanning tree, which is also minimum bottleneck spanning tree
#     """
#     Tree = nx.minimum_spanning_tree(Agent_graph.to_undirected())
#     #print(dir(Tree))
#     #print(Tree.nodes)
#     #print(Tree.edges,"old")
#     return Tree

def generate_minimumspanning_tree_edmonds(Agent_graph):
    """
    :param Agent_graph: Networkx graph for the targets
    :type Agent_graph: Networkx undirected graph
    :return:
    Tree: minimum spanning tree, which is also minimum bottleneck spanning tree
    """
    G=nx.algorithms.tree.branchings.Edmonds(Agent_graph)

    Tree = G.find_optimum(attr="weight",kind="min",style="arborescence")
    #print(Tree.edges)
    #print(Tree)
    return Tree

#Compute the bottleneck cost of the overall path
def compute_cost(Agent_graph,tour):
    """
    :param Agent_graph: Networkx graph for the targets
    :type Agent_graph: Networkx undirected graph
    :param tour: Nodes of the Eulerian tour
    :type tour: python list
    :return:
    cost: Bottleneck cost of the Eulerian tour
    """
    cost=0
    #Go over the path
    for i in range(len(tour)-1):
        # update the cost if the capacity is higher than the max capacity so far
        if Agent_graph[tour[i]][tour[i+1]]['weight']>cost:

            cost=Agent_graph[tour[i]][tour[i+1]]['weight']
    return cost
#Compute the bottleneck cost of the path assignments
def compute_cost_assignments(Agent_graph,tour):
    """
    :param Agent_graph: Networkx graph for the targets
    :type Agent_graph: Networkx undirected graph
    :param tour: Nodes of the Eulerian tour
    :type tour: python list
    :return:
    cost: Bottleneck cost of the Eulerian tour
    """

    cost=[0 for i in range(len(tour))]
    #Go over the tours for each agent
    for i in range(len(tour)):
        for j in range(len(tour[i])-1):
            #print(tour[i][j],tour[i][j+1],tour[i],cost,Agent_graph[tour[i][j]][tour[i][j+1]]['weight'])
            #update the cost if the capacity is higher than the max capacity so far
            if Agent_graph[tour[i][j]][tour[i][j+1]]['weight']>cost[i]:
                cost[i]=Agent_graph[tour[i][j]][tour[i][j+1]]['weight']
            #print(cost,[tour[i][j]],[tour[i][j+1]],tour,Agent_graph[tour[i][j]][tour[i][j+1]]['weight'],Agent_graph[tour[i][j+1]][tour[i][j]]['weight'])

    return cost


def augment_matching(matching,agent_lists,init_state,Bottleneckgraph):
    """
    :param matching: optimal bottleneck matching between agents and assignments
    :param agent_lists: assignments to the each agents
    :param init_state: set of initial states for agent
    :param Bottleneckgraph: networkx graph between initial states of the agent and initial elements of the targets
    :return:
    assignments: set of assignments for each agent
    costs: cost of the assignments
    """
    assignments=[]
    costs=[]
    #go over agents
    #print(matching)
    print("matching now")
    print(init_state,matching,agent_lists)
    for i in range(len(init_state)):
        matchingflag=True
        for item in matching.items():
            # if matching found
            if item[0]==init_state[i]:
                if len(agent_lists[-1*item[1]-1])>0:
                    matchingflag=False

                #add weights if there is a path
                assignments.append(agent_lists[-1*item[1]-1])
                if Bottleneckgraph[item[0]][item[1]]['weight']>0:
                    costs.append(Bottleneckgraph[item[0]][item[1]]['weight'])
                else:
                    costs.append(0)
        print(assignments,matchingflag)

        if matchingflag:
            assignments.append([])
            costs.append(0)

        print(assignments,matchingflag)


    #add empty assignment if agents are not necessary
    while len(assignments)<len(agent_lists):
        assignments.append([])
    #print(costs)
    return(assignments,costs)


def bottleneckassignment2(Bottleneckgraph):
    """
    :param Bottleneckgraph: networkx graph between initial states of the agent and initial elements of the targets
    :return:
    matching: optimal bottleneck matching between agents and assignments
    """
    costlow=0
    costhigh=1e6
    #compute optimal bottleneck matching
    while costhigh-costlow>1e-6:
        aux_graph=Bottleneckgraph.copy()
        cost_bisec=(costhigh+costlow)/2

        #remove edges if the cost is larger than the current cost
        for item in aux_graph.edges:
            if aux_graph[item[0]][item[1]]['weight']>cost_bisec:
                aux_graph.remove_edge(item[0],item[1])
        #try if there exists a matching, if not, do not return it
        try:
            matching=nx.bipartite.maximum_matching(aux_graph)
            costhigh=cost_bisec
            #print(matching,cost_bisec)

        except nx.exception.AmbiguousSolution:
            costlow=cost_bisec

    return matching

def bottleneckassignment(Bottleneckgraph):
    """
    :param Bottleneckgraph: networkx graph between initial states of the agent and initial elements of the targets
    :return:
    matching: optimal bottleneck matching between agents and assignments
    """
    costlow=0
    costhigh=1e6
    #compute optimal bottleneck matching
    print("computing bottleneck assignment")
    while costhigh-costlow>1e-3:
        aux_graph=Bottleneckgraph.copy()
        cost_bisec=(costhigh+costlow)/2

        #remove edges if the cost is larger than the current cost

        dict_item=list(aux_graph.edges(data=True))
        for item in dict_item:
            if aux_graph[item[0]][item[1]]['weight']>cost_bisec:
                aux_graph.remove_edge(item[0],item[1])

        #print(aux_graph.edges)
        #for item in aux_graph.edges:
        #    print(item,aux_graph[item[0]][item[1]]['weight'])
        #try if there exists a matching, if not, do not return it
        matching = nx.maximal_matching(aux_graph)
        matching_node=[]
        for item in list(matching):
            matching_node.append(item[0])
            matching_node.append(item[1])

        matchflag=True
        for nodes in aux_graph:
            #print(nodes,matching,list(matching),nodes in matching_node,matching_node)
            if nodes in matching_node:
                pass
            else:
                matchflag=False
        if matchflag:
            #matching=nx.maximal_matching(aux_graph)
            costhigh=cost_bisec
            #print(matching,cost_bisec,"matching")
            matchingdict=dict()
            for item in list(matching):
                matchingdict[item[0]]=item[1]
                matchingdict[item[1]]=item[0]

        else:
            costlow=cost_bisec

    return matchingdict


def tarjan_scc_init(Graph_cost,num_agent,targets):
    """
    :param Graph_cost: networkx graph between initial states of the agent and initial elements of the targets
    :param num_agent: number of agents
    :return:
    saveitem: assignments for each agent with minimal cots
    costsitem: bottleneck costs for each assignment
    """
    costlow=0
    costhigh=1e4
    saveitem=[]

    #max_edge=max(dict(Graph_cost.edges).items(), key=lambda x: x[1]['weight'])
    # print(max_edge)
    # print(dir(max_edge))
    # print(max_edge[0][0])
    #Graph_cost.remove_edge(max_edge[0][0],max_edge[0][1])
    #computes the set of strongly connected component
    scc_set=list(sorted(nx.strongly_connected_components(Graph_cost), key=len, reverse=True))


    #convert the set into a list for future computations
    scc_list=[[] for i in range(len(scc_set))]
    for i in  range(len(scc_set)):

        if len(scc_set[i])==1:
            singleitem=next(iter(scc_set[i]))
            scc_list[i].append(singleitem)
        else:
            for item in (scc_set[i]):

                scc_list[i].append(item)




    #increase threshold if more SCC's then number of agents
    #print(scc_list,cost_bisec)
    print(scc_list)
    if len(scc_list)>num_agent:
        saveitem=[]
        pathsitem=[]
        for item in targets:

            saveitem.append(item)
        while len(saveitem)<num_agent:
            saveitem.append(list([]))
        for item in targets:

            pathsitem.append(item)
        while len(saveitem)<num_agent:
            pathsitem.append(list([]))
        costsitem=[1e10 for _ in range(num_agent)]
        #costlow=cost_bisec
    else:
        #costhigh=cost_bisec
        saveitem=[]
        pathsitem=[]
        costsitem=[0 for _ in range(num_agent)]
        items_save = [[] for _ in range(num_agent)]
        paths_save = [[] for _ in range(num_agent)]

        #iterate over the path
        for i in range(len(scc_list)):
            #print(scc_list[i])
            #if the length is greater or equal to 2
            if len(scc_list[i])>=2:
                aux_graph2 = Graph_cost.copy()
                #remove unused nodes from the graph
                for node in Graph_cost.nodes:
                    if not node in scc_list[i]:
                        aux_graph2.remove_node(node)
                max_edge = max(dict(aux_graph2.edges).items(), key=lambda x: x[1]['weight'])
                print(max_edge,scc_list[i])
                print(aux_graph2[max_edge[0][0]][max_edge[0][1]]['weight'])
                costsitem[i]=aux_graph2[max_edge[0][0]][max_edge[0][1]]['weight']

                #computes a path in the SCC, needs to be checked
                dfs_path = list(nx.dfs_edges(aux_graph2))
                # #add elements to the path
                for j in range(len(dfs_path)):
                    if not dfs_path[j][0] in items_save[i]:
                        items_save[i].append(dfs_path[j][0])
                    if not dfs_path[j][1] in items_save[i]:
                        items_save[i].append(dfs_path[j][1])
                # #update cost of the path
                for j in range(len(items_save[i])-1):
                    #print(items_save[i][j],items_save[i][j+1])
                    path=nx.dijkstra_path(aux_graph2,items_save[i][j],items_save[i][j+1])
                    #print(path,"path")
                    for k  in range(len(path)-1):
                        paths_save[i].append(path[k])
                #print(paths_save[i],"paths")
                # for k in range(len(paths_save[i])-1):
                #     if Graph_cost[paths_save[i][k]][paths_save[i][k+1]]['weight']>costsitem[i]:
                #        costsitem[i]=Graph_cost[paths_save[i][k]][paths_save[i][k+1]]['weight']
                # try:
                #     if Graph_cost[paths_save[i][len(paths_save[i])-1]][paths_save[i][0]]['weight'] > costsitem[i]:
                #         costsitem[i] = Graph_cost[paths_save[i][len(paths_save[i])-1]][paths_save[i][0]]['weight']
                # except KeyError:
                #     pass
            #add the element if the scc is singular
            elif len(scc_list[i])==1:
                paths_save[i].append(scc_list[i][0])
                items_save[i].append(scc_list[i][0])

        #save the best elements
        for item in items_save:

            saveitem.append(item)
        while len(saveitem)<num_agent:
            saveitem.append(list([]))
        while len(costsitem)<num_agent:
            costsitem.append(0)
        for item in paths_save:
            pathsitem.append(item)
        while len(pathsitem)<num_agent:
            pathsitem.append(list([]))
    #return elements
    #print(saveitem,costsitem)

    return Graph_cost,saveitem,costsitem,pathsitem


def tarjan_scc(Graph_cost,num_agent,targets):
    """
    :param Graph_cost: networkx graph between initial states of the agent and initial elements of the targets
    :param num_agent: number of agents
    :return:
    saveitem: assignments for each agent with minimal cots
    costsitem: bottleneck costs for each assignment
    """
    costlow=0
    costhigh=1e4
    saveitem=[]

    #max_edge=max(dict(Graph_cost.edges).items(), key=lambda x: x[1]['weight'])
    # print(max_edge)
    # print(dir(max_edge))
    # print(max_edge[0][0])
    #Graph_cost.remove_edge(max_edge[0][0],max_edge[0][1])
    #computes the set of strongly connected component
    scc_set=list(sorted(nx.strongly_connected_components(Graph_cost), key=len, reverse=True))
    scc_set_init=len(scc_set)
    savegraph=Graph_cost

    while costhigh-costlow>1e-3:
        aux_graph=Graph_cost.copy()
        cost_bisec=(costhigh+costlow)/2
        #print(costlow,costhigh)
        #remove edges if the cost is larger than the current cost
        dict_item=list(aux_graph.edges(data=True))
        for item in dict_item:
            if aux_graph[item[0]][item[1]]['weight']>cost_bisec:
                aux_graph.remove_edge(item[0],item[1])
        scc_set_aux = list(sorted(nx.strongly_connected_components(aux_graph), key=len, reverse=True))
        #print(scc_set_aux,"scc_set_aux")
        if len(scc_set_aux)>scc_set_init:
            costlow=cost_bisec
            savegraph=aux_graph
        else:
            costhigh=cost_bisec

    #print(dict(Graph_cost.edges))
    Graph_cost=savegraph
    scc_set=list(sorted(nx.strongly_connected_components(Graph_cost), key=len, reverse=True))

    #convert the set into a list for future computations
    scc_list=[[] for i in range(len(scc_set))]
    for i in  range(len(scc_set)):

        if len(scc_set[i])==1:
            singleitem=next(iter(scc_set[i]))
            scc_list[i].append(singleitem)
        else:
            for item in (scc_set[i]):

                scc_list[i].append(item)




    #increase threshold if more SCC's then number of agents
    #print(scc_list,cost_bisec)
    print(scc_list)
    if len(scc_list)>num_agent:
        saveitem=[]
        pathsitem=[]
        for item in targets:

            saveitem.append(item)
        while len(saveitem)<num_agent:
            saveitem.append(list([]))
        for item in targets:

            pathsitem.append(item)
        while len(saveitem)<num_agent:
            pathsitem.append(list([]))
        costsitem=[1e10 for _ in range(num_agent)]
        #costlow=cost_bisec
    else:
        #costhigh=cost_bisec
        saveitem=[]
        pathsitem=[]
        costsitem=[0 for _ in range(num_agent)]
        items_save = [[] for _ in range(num_agent)]
        paths_save = [[] for _ in range(num_agent)]

        #iterate over the path
        for i in range(len(scc_list)):
            #print(scc_list[i])
            #if the length is greater or equal to 2
            if len(scc_list[i])>=2:
                aux_graph2 = Graph_cost.copy()
                #remove unused nodes from the graph
                for node in Graph_cost.nodes:
                    if not node in scc_list[i]:
                        aux_graph2.remove_node(node)
                max_edge = max(dict(aux_graph2.edges).items(), key=lambda x: x[1]['weight'])
                print(max_edge,scc_list[i])
                print(aux_graph2[max_edge[0][0]][max_edge[0][1]]['weight'])
                costsitem[i]=aux_graph2[max_edge[0][0]][max_edge[0][1]]['weight']

                #computes a path in the SCC, needs to be checked
                dfs_path = list(nx.dfs_edges(aux_graph2))
                # #add elements to the path
                for j in range(len(dfs_path)):
                    if not dfs_path[j][0] in items_save[i]:
                        items_save[i].append(dfs_path[j][0])
                    if not dfs_path[j][1] in items_save[i]:
                        items_save[i].append(dfs_path[j][1])
                # #update cost of the path
                for j in range(len(items_save[i])-1):
                    #print(items_save[i][j],items_save[i][j+1])
                    path=nx.dijkstra_path(aux_graph2,items_save[i][j],items_save[i][j+1])
                    #print(path,"path")
                    for k  in range(len(path)-1):
                        paths_save[i].append(path[k])
                #print(paths_save[i],"paths")
                # for k in range(len(paths_save[i])-1):
                #     if Graph_cost[paths_save[i][k]][paths_save[i][k+1]]['weight']>costsitem[i]:
                #        costsitem[i]=Graph_cost[paths_save[i][k]][paths_save[i][k+1]]['weight']
                # try:
                #     if Graph_cost[paths_save[i][len(paths_save[i])-1]][paths_save[i][0]]['weight'] > costsitem[i]:
                #         costsitem[i] = Graph_cost[paths_save[i][len(paths_save[i])-1]][paths_save[i][0]]['weight']
                # except KeyError:
                #     pass
            #add the element if the scc is singular
            elif len(scc_list[i])==1:
                paths_save[i].append(scc_list[i][0])
                items_save[i].append(scc_list[i][0])

        #save the best elements
        for item in items_save:

            saveitem.append(item)
        while len(saveitem)<num_agent:
            saveitem.append(list([]))
        while len(costsitem)<num_agent:
            costsitem.append(0)
        for item in paths_save:
            pathsitem.append(item)
        while len(pathsitem)<num_agent:
            pathsitem.append(list([]))
    #return elements
    #print(saveitem,costsitem)

    return Graph_cost,saveitem,costsitem,pathsitem