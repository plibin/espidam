import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def plot_ODE(modelstate, title, ac=None):
    ac = f"_{ac}" if ac else ""

    plt.figure()
    
    plt.plot(modelstate[f"S{ac}"], label=f"S{ac}", color="blue")
    plt.plot(modelstate[f"I{ac}"], label=f"I{ac}", color="red")
    plt.plot(modelstate[f"R{ac}"], label=f"R{ac}", color="green")

    plt.legend()
    
    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")

    plt.title(title)

    plt.show()

def plot_ODE_age(modelstate, title, pop):
    plt.figure()

    plt.plot(modelstate[f"I_c"], label=f"I_c", color="red")
    plt.plot(modelstate[f"I_a"], label=f"I_a", color="blue")

    plt.legend()

    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")
    plt.ylim(0, pop)

    plt.title(title)

    plt.show()

def plot_ODE_R0s(results, title, ac=None):
    ac = f"_{ac}" if ac else ""

    plt.figure()

    for (R0, modelstate) in results.items():
        ms = results[R0]
        plt.plot(modelstate[f"I{ac}"], label=f"I{ac}_R0_{R0}")
        
    plt.legend()
    
    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")

    plt.title(title)

    plt.show()

def plot_binom_R0s(results, title, ac=None):
    ac = f"_{ac}" if ac else ""

    plt.figure()

    for (R0, modelstate) in results.items():
        
        mean_i = np.mean(modelstate[f"I{ac}"], axis=0)
        lower_95_i = np.percentile(modelstate[f"I{ac}"], 2.5, axis=0)
        upper_95_i = np.percentile(modelstate[f"I{ac}"], 97.5, axis=0)

        plt.plot(mean_i, label=f"I{ac}_R0_{R0}")
        plt.fill_between(range(len(mean_i)), lower_95_i, upper_95_i, alpha=0.3)
        
    plt.legend()
    
    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")

    plt.title(title)

    plt.show()

def plot_binom(modelstates, title, ac=None):
    ac = f"_{ac}" if ac else ""

    plt.figure()

    for iteration in range(len(modelstates[f"S{ac}"])):
        plt.plot(modelstates[f"S{ac}"][iteration], color="blue", alpha=0.1)
        plt.plot(modelstates[f"I{ac}"][iteration], color="red", alpha=0.1)
        plt.plot(modelstates[f"R{ac}"][iteration], color="green", alpha=0.1)

    mean_s = np.mean(modelstates[f"S{ac}"], axis=0)
    mean_i = np.mean(modelstates[f"I{ac}"], axis=0)
    mean_r = np.mean(modelstates[f"R{ac}"], axis=0)

    plt.plot(mean_s, label=f"S{ac}", color="blue")
    plt.plot(mean_i, label=f"I{ac}", color="red")
    plt.plot(mean_r, label=f"R{ac}", color="green")

    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")

    plt.legend()

    plt.title(title)

    plt.show()

def plot_binom_age(modelstates, title, pop):
    plt.figure()

    for iteration in range(len(modelstates[f"I_c"])):
        plt.plot(modelstates[f"I_c"][iteration], color="red", alpha=0.1)
        plt.plot(modelstates[f"I_a"][iteration], color="blue", alpha=0.1)

    mean_c = np.mean(modelstates[f"I_c"], axis=0)
    mean_a = np.mean(modelstates[f"I_a"], axis=0)

    plt.plot(mean_c, label=f"I_c", color="red")
    plt.plot(mean_a, label=f"I_a", color="blue")

    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")
    plt.ylim(0, pop)

    plt.legend()

    plt.title(title)

    plt.show()

def plot_ODE_w_vacc(modelstate, title, pop):
    plt.figure()
    
    plt.plot(modelstate[f"I_a"], label=f"I_a", color="blue")
    plt.plot(modelstate[f"IV_a"], label=f"IV_a", color="turquoise")
    plt.plot(modelstate[f"I_c"], label=f"I_c", color="red")
    plt.plot(modelstate[f"IV_c"], label=f"IV_c", color="orange")
    
    plt.legend()

    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")
    plt.ylim(0, pop)

    plt.title(title)

    plt.show()

def plot_binom_w_vacc(modelstates, title, pop):
    plt.figure()

    for iteration in range(len(modelstates[f"I_c"])):
        plt.plot(modelstates[f"I_c"][iteration], color="red", alpha=0.1)
        plt.plot(modelstates[f"IV_c"][iteration], color="orange", alpha=0.1)   
        plt.plot(modelstates[f"I_a"][iteration], color="blue", alpha=0.1)
        plt.plot(modelstates[f"IV_a"][iteration], color="turquoise", alpha=0.1)

    mean_i_c = np.mean(modelstates[f"I_c"], axis=0)
    mean_iv_c = np.mean(modelstates[f"IV_c"], axis=0)
    mean_i_a = np.mean(modelstates[f"I_a"], axis=0)
    mean_iv_a = np.mean(modelstates[f"IV_a"], axis=0)

    plt.plot(mean_i_c, label=f"I_c", color="red")
    plt.plot(mean_iv_c, label=f"IV_c", color="orange")
    plt.plot(mean_i_a, label=f"I_a", color="blue")
    plt.plot(mean_iv_a, label=f"IV_a", color="turquoise")

    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")
    plt.ylim(0, pop)
    
    plt.legend()

    plt.title(title)

    plt.show()

def rate_to_p(rate, dt):
    return 1 - math.exp(-rate * dt)
