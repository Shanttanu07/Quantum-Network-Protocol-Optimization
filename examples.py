import random
"""
This file contains the demonstration examples,
for the computation of time and Werner parameters
as well as for optimizing cutoff.
Run the example functions will take typically a few minutes.
The three existing examples are:

swap_protocol:
    A nested swap protocol of level 3, with cutoff for each level.

mixed_protocol:
    A mixed protocol with both swap and distillation,
    where the numbers of segments and qubits are not a power of 2.

optimize_cutoff_time:
    Optimization of cutoff for nested swap protocols.

One can run the script directly by opening a console and typing
'python3 examples.py' or open the folder in a Python IDE.
Notice that this file must be kept in the same folder
as the rest of Python scripts.
To choose which example to run, please change the function name
in the last line of this script.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from repeater_algorithm import RepeaterChainSimulation, repeater_sim, plot_algorithm
from repeater_mc import repeater_mc, plot_mc_simulation
from optimize_cutoff import CutoffOptimizer
from logging_utilities import (
    log_init, log_params, log_finish, create_iter_kwargs, save_data)
from utility_functions import secret_key_rate
def simultaneous_mixed_protocols(parameters):
    """
    Runs the simultaneous mixed protocols with various combinations of cut-offs.
    This function accepts parameters including mt_cut, w_cut, rt_cut for each nesting level,
    and N_memory, and returns the calculated secret key rate.
    
    Args:
        parameters (dict): Dictionary containing protocol parameters, including:
            - "p_gen" (float): Success probability of entanglement generation.
            - "p_swap" (float): Success probability of entanglement swapping.
            - "w0" (float): Initial Werner parameter.
            - "t_coh" (int): Coherence time for quantum memory.
            - "t_trunc" (int): Truncation time.
            - "N_memory" (int): Number of quantum memories per node.
            - "mt_cut", "w_cut", "rt_cut" (list): Cut-off parameters for each nesting level.

    Returns:
        float: The calculated secret key rate for the configuration.
    """
    simulator = RepeaterChainSimulation()
    
    # Extract parameters for cut-offs and memory settings
    p_gen = parameters["p_gen"]
    p_swap = parameters["p_swap"]
    w0 = parameters["w0"]
    t_coh = parameters["t_coh"]
    t_trunc = parameters["t_trunc"]
    
    # Lists of cut-offs for each nesting level
    mt_cut = parameters["mt_cut"]
    w_cut = parameters["w_cut"]
    rt_cut = parameters["rt_cut"]
    # print(mt_cut, w_cut, rt_cut)
    
    ############################
    # Part 1: Generate initial entanglement links across all qubit pairs at each repeater link
    pmf_span1_dist1 = np.concatenate(
        (np.array([0.]), p_gen * (1 - p_gen) ** (np.arange(1, t_trunc) - 1))
    )
    w_span1_dist1 = w0 * np.ones(t_trunc)  # Initial Werner parameter

    ############################
    # Part 2: Simultaneous distillation and swapping across repeaters with custom cut-offs
    # Distillation and swapping across links A-B, B-C, C-D with respective cut-offs at each level
    pmf_distilled_AB, w_distilled_AB = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[0], "w_cut": w_cut[0], "rt_cut": rt_cut[0]},
        pmf_span1_dist1, w_span1_dist1, unit_kind="dist"
    )
    pmf_distilled_BC, w_distilled_BC = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[1], "w_cut": w_cut[1], "rt_cut": rt_cut[1]},
        pmf_span1_dist1, w_span1_dist1, unit_kind="dist"
    )
    pmf_distilled_CD, w_distilled_CD = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[2], "w_cut": w_cut[2], "rt_cut": rt_cut[2]},
        pmf_span1_dist1, w_span1_dist1, unit_kind="dist"
    )

    # Swapping concurrently with distillation across all pairs
    pmf_swap_AB, w_swap_AB = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[0], "w_cut": w_cut[0], "rt_cut": rt_cut[0]},
        pmf_distilled_AB, w_distilled_AB, unit_kind="swap"
    )
    pmf_swap_BC, w_swap_BC = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[1], "w_cut": w_cut[1], "rt_cut": rt_cut[1]},
        pmf_distilled_BC, w_distilled_BC, unit_kind="swap"
    )
    pmf_swap_CD, w_swap_CD = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[2], "w_cut": w_cut[2], "rt_cut": rt_cut[2]},
        pmf_distilled_CD, w_distilled_CD, unit_kind="swap"
    )

    ############################
    # Part 3: Final distillation between adjacent repeaters using swapped links
    pmf_final_dist_AB, w_final_dist_AB = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[1], "w_cut": w_cut[1], "rt_cut": rt_cut[1]},
        pmf_swap_AB, w_swap_AB, unit_kind="dist"
    )
    pmf_final_dist_CD, w_final_dist_CD = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[2], "w_cut": w_cut[2], "rt_cut": rt_cut[2]},
        pmf_swap_CD, w_swap_CD, unit_kind="dist"
    )
    # a=random.randint(10000, 100000)/100000
    # Final swap to connect A-B and B-D links
    pmf_final_swap_AD, w_final_swap_AD = simulator.compute_unit(
        {**parameters, "mt_cut": mt_cut[0], "w_cut": w_cut[0], "rt_cut": rt_cut[0]},
        pmf_final_dist_AB, w_final_dist_AB,
        pmf_final_dist_CD, w_final_dist_CD, unit_kind="swap"
    )

    # Calculate and return the secret key rate
    secret_key_rate_value = secret_key_rate(pmf_final_swap_AD, w_final_swap_AD)
    print(f"Calculated Secret Key Rate: {secret_key_rate_value}")
    return secret_key_rate_value







def swap_protocol():
    """
    This example is a simplified version of fig.4 from the paper.
    It calculates the waiting time distribution and the Werner parameter
    with the algorithm shown in the paper.
    A Monte Carlo algorithm is used for comparison.
    """
    parameters = {
        # A protocol is represented by a tuple of 0 and 1,
        # where 0 stands for swap and 1 stands for distillation.
        # This example is a 3-level swap protocol,
        # spanning over 9 nodes (i.e. 8 segments)
        "protocol": (0, 0, 0),
        # success probability of entanglement generation
        "p_gen": 0.1,
        # success probability of entanglement swap
        "p_swap": 0.5,
        # initial Werner parameter
        "w0": 0.98,
        # memory cut-off time
        "cutoff": (16, 31, 55),
        # the memory coherence time,
        # in the unit of one attempt of elementary link generation.
        "t_coh": 400,
        # truncation time for the repeater scheme.
        # It should be increased to cover more time step
        # if the success proability decreases.
        # Commercial hardware can easily handle up to t_trunc=1e5
        "t_trunc": 3000,
        # the type of cut-off
        "cut_type": "memory_time",
        # the sample size for the MC algorithm
        "sample_size": 1000000,
        }
    # initialize the logging system
    log_init("sim", level=logging.INFO)
    fig, axs = plt.subplots(2, 2)

    # Monte Carlo simulation
    print("Monte Carlo simulation")
    t_sample_list = []
    w_sample_list = []
    start = time.time()
    # Run the MC simulation
    t_samples, w_samples = repeater_mc(parameters)
    t_sample_list.append(t_samples)
    w_sample_list.append(w_samples)
    end = time.time()
    print("Elapse time\n", end-start)
    print()
    plot_mc_simulation(
        [t_sample_list, w_sample_list], axs,
        parameters=parameters, bin_width=1, t_trunc=2000)

    # Algorithm presented in the paper
    print("Deterministic algorithm")
    start = time.time()
    # Run the calculation
    pmf, w_func = repeater_sim(parameters)
    end = time.time()
    t = 0
    # Remove unstable Werner parameter,
    # because the the probability mass is too low 10^(-22)
    while(pmf[t] < 1.0e-17):
        w_func[t] = np.nan
        t += 1
    print("Elapse time\n", end-start)
    print()
    plot_algorithm(pmf, w_func, axs, t_trunc=2000)
    print("secret key rate", secret_key_rate(pmf, w_func))

    # plot
    legend = None
    axs[0][0].set_title("CDF")
    axs[0][1].set_title("PMF")
    axs[1][0].set_title("Werner")
    if legend is not None:
        for i in range(2):
            for j in range(2):
                axs[i][j].legend(legend)
    plt.tight_layout()
    fig.show()


def mixed_protocol():
    """
    Here we show a mixed protocol with the number of qubits and segments not
    a power of 2. Notice that it is only for demonstration purpose
    and the protocol is not optimal.
    Setup:
        A four nodes (ABCD) repeater chain with three segments.
        A and D as end nodes each has 3 qubits;
        B and C as repeater nodes each has 6 qubits.
    The name of entangled pairs following the convention:
    span<N>_dist<d>,
    where N is the number of segments this entanglement spans
    and d is the number of elementary links used in the distillation.
    E.g. an elementary link has the name "span1_dist1", while
    the distilled state of two elementary links the name "span1_dist2".
    """
    parameters = {
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.85,
        "t_coh": 400,
        "t_trunc": 3000,
        }
    p_gen = parameters["p_gen"]
    t_trunc = parameters["t_trunc"]
    w0 = parameters["w0"]

    simulator = RepeaterChainSimulation()
    ############################
    # Part1
    # Generate entanglement link for all qubits pairs between AB, BC and CD
    pmf_span1_dist1 = np.concatenate(
        (np.array([0.]),  # generation needs at least one step.
        p_gen * (1 - p_gen)**(np.arange(1, t_trunc) - 1))
        )
    w_span1_dist1 = w0 * np.ones(t_trunc)  # initial werner parameter

    ############################
    # Part2: Between A and B, we distill the entanglement twice.
    # We first distill A1-B1 and A2-B2, save the result in A1-B1
    pmf_span1_dist2, w_span1_dist2 = simulator.compute_unit(
        parameters, pmf_span1_dist1, w_span1_dist1, unit_kind="dist")

    # We then distill A1-B1 and A3-B3, obtain a single link A-B
    pmf_span1_dist3, w_span1_dist3 = simulator.compute_unit(
        parameters, pmf_span1_dist2, w_span1_dist2,
        pmf_span1_dist1, w_span1_dist1, unit_kind="dist")

    ############################
    # Part3: Among B, C and D. Performed simultaneously as part2
    # We first connect all elementary links between B-C and C-D, and then distill.
    # We begin from swap between B-C and C-D, for all 3 pairs of elementary link.
    pmf_span2_dist1, w_span2_dist1 = simulator.compute_unit(
        parameters, pmf_span1_dist1, w_span1_dist1, unit_kind="swap")

    # When B1-D1, B2-D2 and prepared, we distill them
    pmf_span2_dist2, w_span2_dist2 = simulator.compute_unit(
        parameters, pmf_span2_dist1, w_span2_dist1, unit_kind="dist")

    # When B3-D3 is ready, we merge it too with distillation to obtain
    # a single link between B and D
    # Here we add a cutoff on the memory storage time to increase the fidelity,
    # at the cost of a longer waiting time.
    parameters["cutoff"] = 50
    pmf_span2_dist3, w_span2_dist3 = simulator.compute_unit(
        parameters, pmf_span2_dist2, w_span2_dist2,
        pmf_span2_dist1, w_span2_dist1, unit_kind="dist")
    del parameters["cutoff"]

    ############################
    # Part4
    # We connect A-B and B-D with a swap
    parameters["cutoff"] = 50
    pmf_span3_dist3, w_span3_dist3 = simulator.compute_unit(
        parameters, pmf_span1_dist3, w_span1_dist3,
        pmf_span2_dist3, w_span2_dist3, unit_kind="swap")
    del parameters["cutoff"]

    print("secret key rate", secret_key_rate(pmf_span3_dist3, w_span3_dist3))
    # Let's plot the final time and Werner parameter distribution
    fig, axs = plt.subplots(2, 2)
    plot_algorithm(pmf_span3_dist3, w_span3_dist3, axs)
    fig.show()


def optimize_cutoff_time():
    """
    This example includes the optimization of the memory storage cut-off time.
    Without cut-off, this parameters give zero secret rate.
    With the optimized cut-off, the secret key rate can be increased to
    more than 3*10^(-3).
    Depending on the hardware, running the whole example may take a few hours.
    The uniform cut-off optimization is smaller.
    """
    parameters = {
        "protocol": (0, 0, 0),
        "p_gen": 0.1,
        "p_swap": 0.5,
        "w0": 0.98,
        "t_coh": 400,
        "t_trunc": 3000,
        "cut_type": "memory_time",
        }
    log_init("opt", level=logging.INFO)

    # Uniform cut-off optimization. ~ 1-2 min
    logging.info("Uniform cut-off optimization\n")
    # Define optimizer parameters
    opt = CutoffOptimizer(opt_kind="uniform_de", adaptive=True)
    # Run optimization
    best_cutoff_dict = opt.run(parameters)
    # Calculate the secret key rate
    parameters["cutoff"] = best_cutoff_dict["memory_time"]
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate: {:.6f}".format(key_rate))
    del parameters["cutoff"]

    # Nonuniform cut-off optimization.  ~ 5 min
    logging.info("Nonuniform cut-off optimization\n")
    opt = CutoffOptimizer(opt_kind="nonuniform_de", adaptive=True)
    best_cutoff_dict = opt.run(parameters)
    parameters["cutoff"] = best_cutoff_dict["memory_time"]
    pmf, w_func = repeater_sim(parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate: {:.6f}".format(key_rate))

    logging.info("No cut-off\n")
    parameters["cutoff"] = np.iinfo(np.int32).max
    pmf, w_func = repeater_sim(parameters=parameters)
    key_rate = secret_key_rate(pmf, w_func)
    logging.info("Secret key rate without cut-off: {:.6f}".format(key_rate))
    logging.info("Rate without truncation time: {}\n".format(key_rate))
    
def run_protocol_simulation(protocol_index, parameters):
    """
    Runs a specific protocol simulation based on the protocol_index.
    
    Args:
        protocol_index (int): Index specifying which protocol to run.
        parameters (dict): Dictionary containing protocol parameters, including cut-off values.

    Returns:
        float: The calculated secret key rate for the protocol.
    """
    # Run the simultaneous mixed protocols as the main protocol
    if protocol_index == 2:
        # Run `simultaneous_mixed_protocols` and return the `secret_key_rate` directly
        secret_key_rate = simultaneous_mixed_protocols(parameters)
        return secret_key_rate
    else:
        raise ValueError("Invalid protocol index specified for this simulation.")



if __name__ == "__main__":  #  run the example
    mixed_protocol()
