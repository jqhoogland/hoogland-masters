import numpy as np

from randnn import GaussianNN

def test_gaussian_subcritical():
    nn = GaussianNN(coupling_strength=0.8, n_dofs=1000)
    trajectory = nn.run(n_steps=int(1e6), save=False)[::100]
    lyapunov_spectrum = nn.get_lyapunov_spectrum(trajectory, t_ons=10)

    assert np.all(lyapunov_spectrum < 0)

def test_gaussian_chaos():
    nn = GaussianNN(coupling_strength=0.8, n_dofs=1000)
    trajectory = nn.run(n_steps=int(1e6), save=False)[::100]
    lyapunov_spectrum = nn.get_lyapunov_spectrum(trajectory, t_ons=10)

    assert np.all(lyapunov_spectrum[0] > 0)