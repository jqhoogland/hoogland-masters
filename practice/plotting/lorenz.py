import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from pynamics.systems import Lorenz
from pytransfer.transfer_operator import TransferOperator

rc('text', usetex=True)

def plot_lorenz_spectra_with_rates(rates, timestep, n_steps, n_burn_in, n_clusters, rho=28):
    fig, main_ax = plt.subplots()
    right_inset_ax = fig.add_axes([.5, .6, .2, .2])

    for rate in rates:
        dw = Lorenz(rho=rho, timestep=timestep)
        traj = dw.run(n_steps=n_steps, n_burn_in=n_burn_in)
        matrix = TransferOperator(labeling_method="kmeans", n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=rate,k=50)
        main_ax.plot(matrix.eigvals, label=rate)
        right_inset_ax.plot(matrix.eigvals[1:10], label=rate)



    main_ax.set_title(f"Timescale scaling with transition timestep $\\tau$ ($\\rho={rho}$)")
    main_ax.legend(title="$\\tau$")
    main_ax.set_xlabel("$i$")
    main_ax.set_ylabel("$\lambda_i$")

    right_inset_ax.set_xlabel("$i$")
    right_inset_ax.set_ylabel("$\lambda_i$")

def plot_lorenz_spectra_with_rhos(rhos, timestep, n_steps, n_burn_in, n_clusters, rate=1):
    fig, main_ax = plt.subplots()
    right_inset_ax = fig.add_axes([.5, .6, .2, .2])


    for i, rho in enumerate(rhos):
        dw = Lorenz(rho=rho, timestep=timestep)
        traj = dw.run(n_steps=n_steps, n_burn_in=n_burn_in)
        matrix = TransferOperator(labeling_method="kmeans", n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=rate,k=50)
        main_ax.plot(matrix.eigvals, label=rho)
        right_inset_ax.plot(matrix.eigvals[1:10], label=rho)

    main_ax.set_title(f"Timescale scaling with $\\rho$ ($\\tau = {rate}$)")
    main_ax.legend(title="$\\rho$")
    main_ax.set_xlabel("$i$")
    main_ax.set_ylabel("$\lambda_i$")

    right_inset_ax.set_xlabel("$i$")
    right_inset_ax.set_ylabel("$\lambda_i$")


def plot_lorenz_entropies_with_rhos(rhos, rate, timestep, n_steps, n_burn_in, n_clusters):
    fig, main_ax = plt.subplots()

    forward_entropies = np.zeros(len(rhos))
    reverse_entropies = np.zeros(len(rhos))
    entropy_productions = np.zeros(len(rhos))

    for i, rho in enumerate(rhos):
        dw = Lorenz(rho=rho, timestep=timestep)
        traj = dw.run(n_steps=n_steps, n_burn_in=n_burn_in)
        matrix = TransferOperator(labeling_method="kmeans", n_clusters=n_clusters)
        matrix.fit(traj, n_future_timesteps=rate)

        forward_entropies[i] = matrix.forward_entropy
        reverse_entropies[i] = matrix.reverse_entropy
        entropy_productions[i] = matrix.entropy_production

    main_ax.set_title(f"Entropies with $\\rho$")
    main_ax.set_xlabel("$\\rho$")
    main_ax.set_ylabel("$S$")

    main_ax.plot(rhos, forward_entropies, label="Forward entropy")
    main_ax.plot(rhos, reverse_entropies, label="Reverse entropy")
    main_ax.plot(rhos, entropy_productions, label="Entropy production")
    main_ax.legend(title="Type of entropy")
