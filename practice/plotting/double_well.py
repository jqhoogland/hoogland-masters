import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from pynamics.systems import DoubleWell
from pytransfer.transfer_operator import TransferOperator

rc('text', usetex=True)

def plot_eigval_spectra_with_rates(rates, beta, timestep, n_steps, n_burn_in, n_clusters, is_overdamped=True, labeling_method="uniform", inset_pos=[.5, .6]):
    fig, main_ax = plt.subplots()
    right_inset_ax = fig.add_axes([ *inset_pos, .2, .2])

    for rate in rates:
        dw = DoubleWell(beta=beta, timestep=timestep, is_overdamped=is_overdamped)
        traj = dw.run(n_steps=n_steps, n_burn_in=n_burn_in)
        matrix = TransferOperator(labeling_method=labeling_method, n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=rate,k=50)
        main_ax.plot(matrix.eigvals, label=rate)

        right_inset_ax.plot(matrix.eigvals[1:10], label=rate)



    main_ax.set_title(f"Timescale scaling with transition timestep $\\tau$ ($\\beta={beta}$)")
    main_ax.legend(title="$\\tau$")
    main_ax.set_xlabel("$i$")
    main_ax.set_ylabel("$\lambda_i$")

    right_inset_ax.set_xlabel("$i$")
    right_inset_ax.set_ylabel("$\lambda_i$")

def plot_eigval_spectra_with_betas(betas, rate, timestep, n_steps, n_burn_in, n_clusters, is_overdamped=True, labeling_method="uniform"):
    fig, main_ax = plt.subplots()
    right_inset_ax = fig.add_axes([.5, .6, .2, .2])


    for i, beta in enumerate(betas):
        dw = DoubleWell(beta=beta, timestep=timestep, is_overdamped=is_overdamped)
        traj = dw.run(n_steps=n_steps, n_burn_in=n_burn_in)
        matrix = TransferOperator(labeling_method=labeling_method, n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=rate,k=50)
        main_ax.plot(matrix.eigvals, label=beta)
        right_inset_ax.plot(matrix.eigvals[1:10], label=beta)

    main_ax.set_title(f"Timescale scaling with $\\beta$ ($\\tau = {rate}$)")
    main_ax.legend(title="$\\beta$")
    main_ax.set_xlabel("$i$")
    main_ax.set_ylabel("$\lambda_i$")

    right_inset_ax.set_xlabel("$i$")
    right_inset_ax.set_ylabel("$\lambda_i$")

def plot_eigval_spectra_with_downsampling(rates, beta, timestep, n_steps, n_burn_in, n_clusters, is_overdamped=True, labeling_method="uniform"):
    fig, main_ax = plt.subplots()
    right_inset_ax = fig.add_axes([.5, .6, .2, .2])

    for i, rate in enumerate(rates):
        dw = DoubleWell(beta=beta, timestep=timestep, is_overdamped=is_overdamped)
        traj = dw.run(n_steps=n_steps*rate, n_burn_in=n_burn_in)[::rate]
        matrix = TransferOperator(labeling_method=labeling_method, n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=1,k=50)
        main_ax.plot(matrix.eigvals, label=rate)
        right_inset_ax.plot(matrix.eigvals[1:10], label=rate)

    main_ax.set_title(f"Timescale scaling with downsampling rate $R$ ($\\beta={beta}$)")
    main_ax.legend(title="$\\tau$")
    main_ax.set_xlabel("$i$")
    main_ax.set_ylabel("$\lambda_i$")

    right_inset_ax.set_xlabel("$i$")
    right_inset_ax.set_ylabel("$\lambda_i$")



def plot_entropies_with_betas(betas, rate, timestep, n_steps, n_burn_in, n_clusters, is_overdamped=True, labeling_method="uniform"):
    fig, main_ax = plt.subplots()

    forward_entropies = np.zeros(len(betas))
    reverse_entropies = np.zeros(len(betas))
    entropy_productions = np.zeros(len(betas))

    for i, beta in enumerate(betas):
        dw = DoubleWell(beta=beta, timestep=timestep, is_overdamped=is_overdamped)
        traj = dw.run(n_steps=n_steps*rate, n_burn_in=n_burn_in)
        matrix = TransferOperator(labeling_method=labeling_method, n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=rate)

        forward_entropies[i] = matrix.forward_entropy
        reverse_entropies[i] = matrix.reverse_entropy
        entropy_productions[i] = matrix.entropy_production

    main_ax.set_title(f"Entropies with $\\beta$")
    main_ax.set_xlabel("$\\beta$")
    main_ax.set_ylabel("$S$")

    main_ax.plot(betas, forward_entropies, label="Forward entropy")
    main_ax.plot(betas, reverse_entropies, label="Reverse entropy")
    main_ax.plot(betas, entropy_productions, label="Entropy production")
    main_ax.legend(title="Type of entropy")
