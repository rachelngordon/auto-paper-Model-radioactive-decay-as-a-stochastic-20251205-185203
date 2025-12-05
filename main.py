# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_decay_mc(N0, half_life, dt, total_time, n_realizations):
    """Monte‑Carlo simulation of radioactive decay.
    Returns times (1‑D array) and mean remaining atoms (1‑D array)."""
    lam = np.log(2) / half_life
    p = 1 - np.exp(-lam * dt)  # decay probability per atom per step
    n_steps = int(total_time / dt) + 1
    times = np.arange(n_steps) * dt
    # Array to accumulate remaining atoms for each realization
    remaining_all = np.empty((n_realizations, n_steps), dtype=np.int32)
    for r in range(n_realizations):
        N = N0
        remaining = [N]
        for _ in range(1, n_steps):
            # number of decays this step follows Binomial(N, p)
            decays = np.random.binomial(N, p)
            N -= decays
            remaining.append(N)
            if N == 0:
                # fill the rest with zeros
                remaining.extend([0] * (n_steps - len(remaining)))
                break
        remaining_all[r, :] = remaining
    mean_remaining = remaining_all.mean(axis=0)
    return times, mean_remaining

def analytical_solution(N0, half_life, times):
    lam = np.log(2) / half_life
    return N0 * np.exp(-lam * times)

def plot_mc_vs_analytical(times, mean_mc, analytical, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(times, mean_mc, label='Monte Carlo mean', lw=2)
    plt.plot(times, analytical, '--', label='Analytical', lw=2)
    plt.xlabel('Time')
    plt.ylabel('Mean remaining atoms')
    plt.title('Monte Carlo vs Analytical Radioactive Decay')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def simulate_single_atom_decay(half_life, dt, n_trials):
    """Simulate decay times for a single atom over many trials.
    Returns an array of decay times (float)."""
    lam = np.log(2) / half_life
    p = 1 - np.exp(-lam * dt)
    # Geometric distribution: number of trials until first success (including success)
    # numpy's geometric returns number of trials, so time = (k-1)*dt
    steps_until_decay = np.random.geometric(p, size=n_trials)
    times = (steps_until_decay - 1) * dt
    return times

def plot_decay_time_histogram(times, half_life, dt, filename):
    lam = np.log(2) / half_life
    max_time = times.max()
    bins = np.arange(0, max_time + dt, dt)
    plt.figure(figsize=(8, 5))
    plt.hist(times, bins=bins, density=True, alpha=0.6, label='Simulation')
    t_vals = np.linspace(0, max_time, 500)
    plt.plot(t_vals, lam * np.exp(-lam * t_vals), 'r-', lw=2, label='Analytical PDF')
    plt.xlabel('Decay time')
    plt.ylabel('Probability density')
    plt.title('Distribution of Single‑Atom Decay Times')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # Parameters (can be adjusted)
    N0 = 1000            # initial number of atoms
    half_life = 5.0      # half‑life in arbitrary time units
    dt = 0.1             # time step
    total_time = 30.0    # total simulation time
    n_realizations = 5000
    n_trials_single = 200000

    # Experiment 1: Monte Carlo vs analytical
    times, mean_mc = simulate_decay_mc(N0, half_life, dt, total_time, n_realizations)
    analytical = analytical_solution(N0, half_life, times)
    plot_mc_vs_analytical(times, mean_mc, analytical, 'remaining_vs_time.png')

    # Experiment 2: Distribution of single‑atom decay times
    decay_times = simulate_single_atom_decay(half_life, dt, n_trials_single)
    plot_decay_time_histogram(decay_times, half_life, dt, 'decay_time_histogram.png')

    # Primary numeric answer: decay constant lambda
    lam = np.log(2) / half_life
    print('Answer:', lam)

if __name__ == '__main__':
    main()

