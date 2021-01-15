# # Test computation of minimal capacity
import fimdp
from fimdpenv import setup
setup()
from uuvmodel import create_env
from uuvmodel import animate_simulation, calc_exptimetotarget, visualize_multisnapshots
from fimdp.energy_solvers import GoalLeaningES, BasicES
from fimdp.objectives import AS_REACH, BUCHI, MIN_INIT_CONS, POS_REACH, SAFE
#from reachability_examples import ultimate
#from fimdp.energy_solver import AS_REACH
#from fimdp.energy_solver import BUCHI
from fimdp.mincap_solvers import bin_search
import matplotlib.pyplot as plt
from networkx.algorithms import tournament
from fimdp.core import CounterStrategy

#import MarsEnv
import numpy as np
import scipy.optimize
from scipy.optimize import linear_sum_assignment
import networkx as nx
import math
import multi_agent_codes
from fimdpenv import setup, UUVEnv
from fimdpenv.UUVEnv import SynchronousMultiAgentEnv
import random

#from fimdpenv.env_multiagent import setup
#from env_multiagent import visualize_allocation

#from env_multiagent import create_env


gridsize = 40
random.seed(0)
def create_multiagent_env():
    """
    :return:
    e: multi-agent UUVEnv
    consmdp: Underlying consumption MDP of the environment
    targets: set of targets
    num_agents: number of agents
    init_state: set of initial states
    """
    #env_multiagent setup function
    setup()
    #generate environment
    init_state = [545, 95,157]
    targets = [454, 64, 547, 372, 183, 155, 611]

    #targets = [454, 10, 547, 872, 183, 135, 411]
    init_state= [452,664,123]
    targets = [125, 164, 247, 572, 383, 355, 411]
    #targets = [312, 164, 247, 572, 383, 355, 411]

    init_state= [32,74,398]
    targets = [23, 44, 93, 112, 87, 93, 35]
    num_agents=5

    init_state=list(random.sample(range(gridsize*gridsize), num_agents))
    targets=list(random.sample(range(gridsize*gridsize), 60))
    print(init_state)
    #, 136]
        #, 203, 272]
    reload_list=[22,107]
    reload_list=[]
    for item in targets:
        reload_list.append(item)
    # for item in init_state:
    #     reload_list.append(item)
    e = SynchronousMultiAgentEnv(num_agents=num_agents, grid_size=[gridsize, gridsize], capacities=[100 for _ in range(num_agents)], reloads=reload_list,
                                 targets=targets, init_states=init_state,
                                 enhanced_actionspace=0)    #generate consumption MDP
    #generate consumption mdp
    e.create_consmdp()
    consmdp1 = e.get_consmdp()
    consmdp = consmdp1[0]
    #generate targets
    targets = e.targets
    #print(targets)
    return e,consmdp,targets,num_agents,init_state

def create_costs_for_agents(Agent_graph,consmdp,init_state,targets,agent_target):
    """
    :param Agent_graph: networkx graph of with the list of targets
    :param consmdp: consumption MDP
    :param init_state: set of initial states for the agents
    :param targets: set of targets
    :param Agent_target: networkx graph of with the list of agents and targets

    :return:
    Agent_graph: updated networkx graph with the list of targets and capacities
    """
    for item in init_state:

        for item2 in targets:

            #if not item==item2:
                #compute the capacity by bin_search
                #result = bin_search(consmdp, item, item2 ,objective=BUCHI)
            result=np.random.uniform(0,0.1)
            #print(item,item2,result)
            #add the edge with capacity
            result=divmod(item,gridsize)
            result1=divmod(item2,gridsize)
            #print(result,result1,dist_fun(result,result1))
            agent_target.add_edge(item, item2, weight=dist_fun(result,result1)*0.6)

            #result = bin_search(consmdp, item2, item ,objective=BUCHI)
            #result=np.random.uniform(0,0.1)
            agent_target.add_edge(item2, item,  weight=dist_fun(result,result1)*0.6)
            #print(item2,item,result)

    for item in targets:
        for item2 in targets:
            #compute the capacity by bin_search
            #result = bin_search(consmdp, item, item2 ,objective=BUCHI)
            result=np.random.uniform(0,1)
            #if item==item2:
            #    result=0
            #print(item,item2,result)
            #add the edge with capacity
            result=divmod(item,gridsize)
            result1=divmod(item2,gridsize)
            #print(result,result1,dist_fun(result,result1))

            Agent_graph.add_edge(item, item2,  weight=dist_fun(result,result1))

    return Agent_graph,agent_target




def dist_fun(p1,p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 +
                     (p2[1] - p1[1]) ** 2 )


def create_costs_for_agents_targets(consmdp,agent_lists,cost_lists,init_state,agent_target):
    """
    :param consmdp: consumption MDP
    :param agent_lists: assignments to the each agents
    :param cost_lists: list of the cost of each path
    :param init_state: set of initial states
    :return:
    Bottleneckgraph: networkx graph between initial states of the agent and initial elements of the targets
    """

    #generate bipartite graph with the set of initial states and initial targets for each agent
    Bottleneckgraph=nx.Graph()
    dict_costs=dict()
    dict_costs2=dict()

    for i in range(len(init_state)):
        for j in range(len(agent_lists)):
            dict_costs[init_state[i],j]=[1e4,-1]
            dict_costs2[init_state[i],j]=[1e4,-1]

    for i in range(len(init_state)):
        Bottleneckgraph.add_node(init_state[i])

    for j in range(len(agent_lists)):
        Bottleneckgraph.add_node(-j-1)

    #go over initial agent states and initial targets for each agent
    for i in range(len(init_state)):
        for j in range(len(agent_lists)):
            #if target exists
            if len(agent_lists[j]) >= 1:
                #for k in range(len(agent_lists[j])):
                costmax=1e10
                for k in range(len(agent_lists[j])):

                    item1=init_state[i]
                    item2=agent_lists[j][k]
                    #compute the capacity between initial agent states and initial targets
                    #result = bin_search(consmdp, item1, item2, objective=BUCHI)
                    #result = np.random.uniform(0, 1)
                    result = agent_target[item1][item2]['weight']
                    if result<costmax:
                        costmax=result
                        item3=item2
                    #print(item1, item2, result)
                    #item2=agent_lists[j][0]
                    #update the cost if the capacity is higher

                if costmax>cost_lists[j] and costmax <dict_costs[init_state[i],j][0]:
                    dict_costs[init_state[i], j]=[costmax,item3]

                elif costmax<=cost_lists[j] and cost_lists[j] <dict_costs[init_state[i],j][0]:
                    dict_costs[init_state[i], j]=[cost_lists[j],item3]
                costmax=1e10
                for k in range(len(agent_lists[j])):

                    item1=init_state[i]
                    item2=agent_lists[j][k]
                    #compute the capacity between initial agent states and initial targets
                    #result = bin_search(consmdp, item1, item2, objective=BUCHI)
                    result = agent_target[item2][item1]['weight']
                    if result<costmax:
                        costmax=result
                        item3=item2
                    #print(item1, item2, result)
                    #update the cost if the capacity is higher
                if costmax>cost_lists[j] and costmax <dict_costs2[init_state[i],j][0]:
                    dict_costs2[init_state[i], j]=[costmax,item3]
                elif costmax<=cost_lists[j] and cost_lists[j] <dict_costs2[init_state[i],j][0]:
                    dict_costs2[init_state[i], j]=[cost_lists[j],item3]
                #add min of the all edge costs if a transition exist
                if dict_costs2[init_state[i],j][0]<dict_costs[init_state[i],j][0]:
                    Bottleneckgraph.add_edge(init_state[i], -j-1, weight=dict_costs[init_state[i], j][0])
                else:
                    dict_costs[init_state[i], j][0]=dict_costs2[init_state[i],j][0]
                    Bottleneckgraph.add_edge(init_state[i], -j-1, weight=dict_costs[init_state[i], j][0])

            else:
                #add -1 if the path is singular
                Bottleneckgraph.add_edge(init_state[i], -j-1, weight=-1)

    return Bottleneckgraph,dict_costs,dict_costs2




if __name__ == "__main__":


    #create multi agent env
    env,MDP,T,num_agent,init_state=create_multiagent_env()
    #print(MDP,T)
    #generate networkx graph
    Graph,graph_agent_target=multi_agent_codes.generate_Graph(T,init_state)
    #generate capacities
    print("Computing capacities with bin search")
    Graph_cost,graph_agent_target_cost=create_costs_for_agents(Graph,MDP,init_state,T,graph_agent_target)

    print("----------------------------------------")
    print("Min-max-k-cover based algorithm")
    print("----------------------------------------")
    #print(Graph_cost)
    #generate minimum spanning tree
    # Tree=multi_agent_codes.generate_minimumspanning_tree_edmonds(Graph_cost)
    #
    # #print(Tree.edges,"kruskal")
    # #generate approximate hamiltonian path
    # hamilton_path=multi_agent_codes.min_hamilton(Tree)
    # #print(hamilton_path)
    # #compute cost of the path
    # cost=multi_agent_codes.compute_cost(Graph_cost,hamilton_path)
    # #print(cost)
    # #assign paths to the agents
    # print("Computing bisection for path assignments")
    #
    # agent_lists,cost_bisec=multi_agent_codes.bisection_loop(Graph_cost,hamilton_path,num_agent)
    # print("Showing the allocation for agents")
    # print(agent_lists)
    # #print(agent_lists2)
    # #compute the allocation costs
    # cost_lists=multi_agent_codes.compute_cost_assignments(Graph_cost,agent_lists)
    # print("Showing the costs of the allocations")
    # print(cost_lists)
    # #visualize_allocation(e)
    # print("Generating capacities between agents and set of paths with bin search")
    #
    # bottleneck_graph,dict_in,dict_out=create_costs_for_agents_targets(MDP,agent_lists,cost_lists,init_state,graph_agent_target)
    # #allocates targets to agents
    # print("Computing bottleneck assignment")
    # matching=multi_agent_codes.bottleneckassignment(bottleneck_graph)
    # #matching2=multi_agent_codes.tarjan_scc(Graph_cost,num_agent)
    # final_assignments,final_costs=multi_agent_codes.augment_matching(matching,agent_lists,init_state,bottleneck_graph)
    # print("Showing the final ordered allocation for agents with initial states")
    # print(final_assignments)
    # print("Showing the final ordered costs of the allocations")
    # print(final_costs)

    #cost_lists2=multi_agent_codes.compute_cost_assignments(Graph_cost,agent_lists2)
    print("----------------------------------------")
    print("Tarjan-based part")
    print("----------------------------------------")
    #compute the allocation costs using tarjan
    cost_max=1e4
    final_tarjan_assignment=[]
    final_tarjan_cost=[]
    final_in_dict=[]
    final_out_dict=[]
    final_paths=[]
    while True:
        Graph_cost,agent_lists2,cost_lists2,paths_save2 = multi_agent_codes.tarjan_scc(Graph_cost, num_agent,T)
        if max(cost_lists2)>1e8:
            break
        # print("Showing the allocation for agents")
        # print(agent_lists2)
        # print("Showing the costs of the allocations")
        # print(cost_lists2)
        # #visualize_allocation(e)
        # print("Generating capacities between agents and set of paths with bin search")

        bottleneck_graph2,dict_in2,dict_out2=create_costs_for_agents_targets(MDP,agent_lists2,cost_lists2,init_state,graph_agent_target_cost)
        #allocates targets to agents
        #print("Computing bottleneck assignment")
        matching2=multi_agent_codes.bottleneckassignment2(bottleneck_graph2)
        #matching2=multi_agent_codes.tarjan_scc(Graph_cost,num_agent)
        #print(matching2)

        final_assignments2,final_costs2=multi_agent_codes.augment_matching(matching2,agent_lists2,init_state,bottleneck_graph2)
        print(final_costs2,cost_max)
        if max(final_costs2)<=cost_max:
            final_tarjan_assignment=final_assignments2
            final_tarjan_cost=final_costs2
            final_in_dict=dict_in2
            final_out_dict=dict_out2
            final_paths=paths_save2
            cost_max=max(final_costs2)
        # print("Showing the final ordered allocation for agents with initial states")
        # print(final_assignments2)
        # print("Showing the final ordered costs of the allocations")
        # print(final_costs2)
    print("Showing the best ordered allocation for agents with initial states")
    print(final_tarjan_assignment)
    print("Showing the best ordered costs of the allocations")
    print(final_tarjan_cost)
    print("Showing targets to enter the SCCs for each agents")
    print(final_in_dict)
    print(final_out_dict)
    print(final_paths)
    print("init states")
    print(init_state)
    env.allocate_targets(final_tarjan_assignment)
    consmdp1 = env.get_consmdp()
    MDP = consmdp1[0]
    reload_list=list(random.sample(range(gridsize*gridsize), 20))
    #reload_list=T
    num_agents=num_agent
    targets=T
    env = SynchronousMultiAgentEnv(num_agents=num_agents, grid_size=[gridsize, gridsize], capacities=[50 for _ in range(num_agents)], reloads=reload_list,
                                 targets=targets, init_states=init_state,
                                 enhanced_actionspace=0)
    consmdp1 = env.get_consmdp()
    for item in final_tarjan_assignment:
        if len(item)==0:
            final_tarjan_assignment[final_tarjan_assignment.index(item)]=init_state[final_tarjan_assignment.index(item)]
    env.allocate_targets(final_tarjan_assignment)

    MDP = consmdp1[0]
    #generate targets
    #compute strategies, pulled from Pranay
    for agent in env.agents:
        print(agent)
        solver = GoalLeaningES(MDP, env.capacities[agent], env.targets_alloc[agent], threshold=0.3)
        selector = solver.get_selector(AS_REACH)
        strategy = CounterStrategy(env.consmdp, selector, env.capacities[agent], env.energies[agent],
                                   init_state=env.init_states[agent])
        env.update_strategy(strategy, agent)
    env.animate_simulation(num_steps=400, interval=100)