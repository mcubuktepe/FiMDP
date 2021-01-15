import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fimdpenv.UUVEnv import SingleAgentEnv
from fimdp.core import CounterStrategy
from fimdp.energy_solvers import GoalLeaningES, BasicES
from fimdp.objectives import AS_REACH, BUCHI, MIN_INIT_CONS, POS_REACH, SAFE
import matplotlib.animation as animation


# Create test environment configurations
def create_env(env_name, capacity=200, heading_sd=1.624, reloads_input=None):
    """
    Create different environments with different grid sizes, target states, and reload
    states
    """
    if env_name == '2R-1T-simple':
        grid_size = (20,20)
        capacity = capacity 
        init_state = 4*grid_size[0]+2
        reloads = [5*grid_size[0]+5, 12*grid_size[0] - 5]
        targets = [grid_size[0]*grid_size[1] - 3*grid_size[0] - 8]
    elif env_name == '1R-1T-simple':
        grid_size = (20,20)
        capacity = capacity 
        init_state = 5*grid_size[0]+2
        reloads = [6*grid_size[0]+2-12]
        targets = [grid_size[0]*grid_size[1] - 7*grid_size[0] - 8]
    elif env_name == '2R-1T-complex':
        grid_size = (20,20)
        capacity = capacity 
        init_state = 5*grid_size[0]+2
        reloads = [8*grid_size[0]+3, 12*grid_size[0] - 5]
        targets = [grid_size[0]*grid_size[1] - 3*grid_size[0] - 8]
    elif env_name == '4R-5T-complex':
        grid_size = (30,80)
        capacity = capacity 
        init_state = 15*grid_size[1]+ 10
        reloads = [5*grid_size[1]+20, 5*grid_size[1]+60, 23*grid_size[1]+10, 23*grid_size[1]+70]
        targets = [5*grid_size[1]+10, 5*grid_size[1]+40, 5*grid_size[1]+70, 27*grid_size[1]+20, 27*grid_size[1]+60]   
    elif env_name == '4R-1T-complex':
        grid_size = (30,80)
        capacity = capacity 
        init_state = 5*grid_size[1]+ 10
        reloads = [5*grid_size[1]+30,5*grid_size[1]+50,17*grid_size[1]+45,20*grid_size[1]+65]
        targets = [27*grid_size[1]+60]   
    else:
        raise Exception("No configuration with that name. Please check the entered name again")
    if reloads_input is None:
        env = SingleAgentEnv(grid_size, capacity, reloads, targets, init_state, heading_sd=heading_sd, enhanced_actionspace=1)
    else:
        env = SingleAgentEnv(grid_size, capacity, reloads_input, targets, init_state, heading_sd=heading_sd)
    return env


def create_counterstrategy(consmdp, capacity, targets, init_state, energy=None, solver=GoalLeaningES, objective=BUCHI, threshold=0.1):
    """
    Create counter strategy for given parameters and the current consMDP object
    and return the strategy
    """
    
    if energy is None:
        energy=capacity
    if solver == GoalLeaningES:
        slvr = GoalLeaningES(consmdp, capacity, targets, threshold=threshold)
    elif solver == BasicES:
        slvr = BasicES(consmdp, capacity, targets)
    selector = slvr.get_selector(objective)
    strategy = CounterStrategy(consmdp, selector, capacity, energy, init_state=init_state)
    return strategy


def animate_simulation(env, num_steps=100, interval=100):
    """
    Execute the strategy for num_steps number of time steps and animate the
    resultant trajectory.
    """
    
    env.reset()
    fig = plt.figure()
    ax = fig.gca()
    ax.axis('off')
    im = plt.imshow(env._states_to_colors(), animated=True)
    plt.close()
    
    im_history = [env._states_to_colors()]
    energy_history = [env.energies[0]]
    
    for i in range(num_steps):
        env.step()
        im_history.append(env._states_to_colors())
        energy_history.append(env.energies[0])
        
    def updatefig(frame_count):
        im.set_array(im_history[frame_count])
        ax.set_title("Agent Energy: {}, Time Steps: {}".format(energy_history[frame_count], frame_count))
        return im
    return animation.FuncAnimation(fig, updatefig, frames=num_steps, interval=interval), im_history, energy_history


def visualize_snapshots(im_history, energy_history, snapshots_indices=[], annotate=True, filename=None):
    """
    Create static subplots showing different instances of a simulation. The 
    number of snaphot indices given must be an even number
    """
    
    num_snapshots = len(snapshots_indices)
    fig, axes = plt.subplots(nrows=2, ncols=num_snapshots//2)

    count = 0
    for ax in axes.flat:
        index = snapshots_indices[count]
        img_data = im_history[index]
        ax.imshow(img_data)
        if annotate is True:    
            name = '('+chr(ord('`')+count+1)+') '+'t = {} e = {}'.format(index, energy_history[index])
            ax.set_xlabel(name)
        else:
            ax.set_xlabel('t = {} e = {}'.format(index, energy_history[index]))
        count += 1
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename+'.png', dpi=400)
        
        
def visualize_multisnapshots(im_history, energy_history, snapshots_indices=[], annotate=True, annotate_names=None, filename=None):
    """
    Create static subplots showing different instances of a simulation. The 
    number of snaphot indices given must be an even number
    """
    
    len_snapshotsindices = [len(x) for x in snapshots_indices]
    num_datasets = len(snapshots_indices)
    num_snapshots = sum(len_snapshotsindices)
    fig, axes = plt.subplots(nrows=num_datasets, ncols=num_snapshots//num_datasets)

    dataset = 0
    count = 0
    for ax in axes.flat:
        index = snapshots_indices[dataset][count]
        img_data = im_history[dataset][index]
        ax.imshow(img_data)
        if annotate is True:    
            if annotate_names is not None:
                name = '('+annotate_names[dataset]+chr(ord('`')+count+1)+') '+'t = {} e = {}'.format(index, energy_history[dataset][index])
            else:
                name = '('+str(dataset+1)+chr(ord('`')+count+1)+') '+'t = {} e = {}'.format(index, energy_history[dataset][index])
            ax.set_xlabel(name, fontsize=8)
        else:
            ax.set_xlabel('t = {} e = {}'.format(index, energy_history[dataset][index]))
        count += 1
        if count == len(snapshots_indices[dataset]):
            dataset += 1
            count = 0
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename+'.pdf', dpi=400)
        
    
def plot_exptimetotarget(env, threshold_list, num_runs=1000, filename='exptime_diffthresholds.csv'):
    """
    Plot the expected time to reach the target for various thresholds in a given 
    test environment
    """
    
    exptime_diffthreshold = {}
    for threshold in threshold_list:
        env.create_counterstrategy(threshold=threshold)
        time_hist = []
        for run in range(num_runs):
            env.reset()
            while True:
                env.step()
                if env.positions[0] in env.targets:
                    time_hist.append(env.num_timesteps)
                    break
        exptime_diffthreshold[threshold] = np.mean(time_hist)
        df = pd.DataFrame({'threshold':list(exptime_diffthreshold.keys()), 'times':list(exptime_diffthreshold.values())})
        df.to_csv(filename)
        sns.scatterplot(x="threshold", y="times", data=df)
        plt.xlabel("threshold in GoalLeaningES")
        plt.ylabel("expected number of time steps")    
        
        
def calc_exptimetotarget(env, num_runs=1000):
    """
    Caclulate the expected time to reach the target for a in a given 
    test environment using the strategy stored in the strategy object of the 
    environment.
    """
    
    if env.strategies is None:
        raise Exception('add a strategy to the environment using the update_strategy() method.')
    
    time_hist = []
    for run in range(num_runs):
        env.reset()
        while True:
            env.step()
            if env.positions[0] in env.targets:
                time_hist.append(env.num_timesteps)
                break
    return np.mean(time_hist)