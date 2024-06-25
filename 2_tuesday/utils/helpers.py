import math
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.signal import savgol_filter
from IPython import display

# Helper object for environment transitions
Transition = namedtuple('Transition', ('state', 
                                       'action', 
                                       'next_state', 
                                       'reward'))

# Helper function for plotting
def plot_performance(cumulative_rewards: list[float], 
                     show_result: bool = False,
                     window_length: int = 25, 
                     polyorder: int = 3) -> None:
    """
    Plots the learning curve, i.e., performance during training.

    Arguments
    ---------
    cumulative_rewards :  The list of cumulative rewards from episodes.
    show_result        :  Flag to either show the final result or the ongoing training plot.

    """
    # Create a new figure
    plt.figure(1)

    # Convert the reward list to a tensor
    rewards_t = torch.tensor(cumulative_rewards, dtype=torch.float)
    
    # Configure plot title based on whether or not we are still training
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    
    # Plot cumulative rewards
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.plot(rewards_t.numpy(), label='Cumulative Reward', alpha=0.25)
    
    # Apply Savitzky-Golay filter to smooth the curve.
    # Smoothing is useful because it helps in identifying trends and patterns
    # that might be obscured by noise and fluctuations in the raw results.
    # (https://en.wikipedia.org/wiki/Savitzky–Golay_filter)
    if len(rewards_t) >= window_length:
        smoothed_rewards = savgol_filter(rewards_t.numpy(), 
                                         window_length=window_length, 
                                         polyorder=polyorder)
        plt.plot(smoothed_rewards, label='Smoothed Cumulative Reward', linewidth=2, color='orange')

    plt.legend()
    plt.pause(0.001)  # pause briefly to update plots
    
    # Display the plot in the notebook
    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())
        plt.ioff()
        plt.show()

# Helper function for calculating epsilon
def calculate_epsilon(EPS_START: float, 
                      EPS_END: float, 
                      EPS_DECAY: float, 
                      steps_done: int) -> float:
    """
    Calculates the current value of epsilon.

    Arguments
    ---------
    EPS_START  :  The starting value of epsilon.
    EPS_END    :  The final value of epsilon.
    EPS_DECAY  :  The decay parameter for epsilon.
    steps_done :  The number of steps done so far.

    Returns
    -------
    epsilon :  The current value of epsilon.

    """
    epsilon = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    return epsilon
