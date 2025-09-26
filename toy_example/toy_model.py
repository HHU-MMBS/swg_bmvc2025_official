import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from tqdm import tqdm


#------------------------------------------
# Math
def t_schedule(num_steps, sigma_min, sigma_max, rho):
    """Time step discretization."""
    step_indices = np.arange(num_steps)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.concatenate([t_steps, np.zeros_like(t_steps[:1])])  # t_N = 0


def log_gaussian_density(x, mu, sigma_sq):
    """
    Compute the log of the Gaussian density for a given mean and variance.
    Args:
        x: The value at which to evaluate the density (n-dimensional vector)
        mu: The means of the Gaussian distribution (m x n matrix where each row is a mean vector)
        sigma_sq: The variance of the Gaussian distribution (scalar, same for all dimensions)

    Returns:
        np.ndarray: The log of the Gaussian density values for each mean vector (m-dimensional vector)
    """
    n = x.shape[0]  # dimensionality of the data
    diff = mu - x  # shape (m, n)
    log_normalization_constant = -0.5 * n * np.log(2 * np.pi * sigma_sq)
    log_exponent = -0.5 * np.sum(diff ** 2, axis=1) / sigma_sq
    return log_normalization_constant + log_exponent


def y_star_log(x, t_sq, data, epsilon=1e-128):
    """
    Optimal prediction (Eq. 13) using log-space for numerical stability.
    Args:
        x: Current point in the trajectory x(t)
        t_sq: Current noise level squared sigma(t)^2
        data: All data points
        epsilon: Small constant to avoid division by zero

    Returns:
        np.ndarray: The optimal prediction y^*
    """

    # Compute log of Gaussian density
    log_gauss = log_gaussian_density(x, data, t_sq)  # [N,] for N datapoints of dimension D
    max_log_gauss = np.max(log_gauss)  # Max log density for numerical stability
    gauss_stable = np.exp(log_gauss - max_log_gauss)  # This avoids underflow in exp
    log_N = max_log_gauss + np.log(np.sum(gauss_stable))
    
    # If N is very small, return the closest datapoint
    if np.exp(log_N) < epsilon:
        print(f"Warning: y^* returning closest datapoint")
        return data[np.abs(data - x).sum(axis=1).argmin()]
    
    weighted_sum = (data * gauss_stable[:, None]).sum(axis=0)
    return weighted_sum / (np.sum(gauss_stable) + epsilon)


def y_toy(x, t, c, data, delta, cond):
    """
    Error prone prediction y_delta.
    Args:
        x: Current point in the trajectory x(t)
        t: Current noise level sigma(t)
        c: Index of the trajectory
        data: All data points
        delta: Uncertainty around the data points
        cond: Conditional model

    Returns:
        np.ndarray: The prediction y_delta
    """
    t_sq, delta_sq = t**2, delta**2
    y = y_star_log(x, t_sq + delta_sq, data) if not cond else data[c % data.shape[0]]  # Optimal model
    return (t_sq * y + x * delta_sq) / (t_sq + delta_sq) if delta != 0 else y  # Model with delta-uncertainty


def compute_score(x, t, i, c, guid_weight, interval, data, pos_kwargs, neg_kwargs):
    """
    Compute the score function at the point x and time t.
    Args:
        x: Current point in the trajectory x(t)
        t: Current noise level sigma(t)
        i: Current time step
        c: Index of the trajectory
        guid_weight: Guidance weight
        interval: Interval where the guidance weight is used
        data: All data points
        pos_kwargs: Positive model parameters
        neg_kwargs: Negative model parameters

    Returns:
        np.ndarray: The score function at x and t
        np.ndarray: The prediction y(t)
        float: The guidance weight
    """
    y_pos = y_toy(x, t, c, data, **pos_kwargs)  # (D,)
    y_neg = y_toy(x, t, c, data, **neg_kwargs)
        
    if guid_weight == 'opt_weight' and interval[0] <= i <= interval[-1]:
        y_opt = y_star_log(x, t**2, data) if not pos_kwargs['cond'] else data[c % data.shape[0]]  # Optimal model
        guid_weight = norm(y_opt - y_pos) / np.maximum(norm(y_pos - y_neg), 1e-128)
        # guid_weight = norm(y_opt - y_pos) / norm(y_pos - y_neg)
    elif interval[0] <= i <= interval[-1]:
        pass
    else:
        guid_weight = 0
    
    y_pred = y_pos + guid_weight * (y_pos - y_neg)
    d_cur = (x - y_pred) / t  # score = force / t^2
    return d_cur, y_pred, guid_weight


def heun_step(data, x_cur, t_cur, t_next, i, c, guid_weight, interval, 
              pos_kwargs, neg_kwargs):
    """
    One Heun-step.
    Args:
        data: All data points
        x_cur: Current point in the trajectory  x(t)
        t_cur: Current noise level sigma(t)
        t_next: Next noise level sigma(t+1)
        i: Current time step
        c: Index of the trajectory
        guid_weight: Guidance weight
        interval: Interval where the guidance weight is used
        pos_kwargs: Positive model parameters
        neg_kwargs: Negative model parameters

    Returns:
        np.ndarray: The next point in the trajectory x(t+1)
        np.ndarray: The score function at x(t)
        np.ndarray: The prediction y(t)
        float: The guidance weight
    """
    score_kwargs = {'data': data, 'i': i, 'c': c, 'guid_weight': guid_weight, 'interval': interval, 
                    'pos_kwargs': pos_kwargs, 'neg_kwargs': neg_kwargs}
    
    # Euler step.
    d_cur, y_pred, guid_weight = compute_score(x_cur, t_cur, **score_kwargs)
    x_eul = x_cur + (t_next - t_cur) * d_cur

#     # Apply 2nd order correction (Heun).
#     if i < num_steps - 1:
#         d_prime, y_pred2, guid_weight2 = compute_score(x_eul, t_next, **score_kwargs)
#         x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
#     else:
#         x_next = x_eul
    x_next = x_eul

    return x_next, d_cur, y_pred, guid_weight


def compute_traject(init, data, n, t_steps, **heun_kwargs):
    """
    Simulate a full trajectory.
    Args:
        init: Initialization mode
        data: All data points
        n: Number of trajectories
        t_steps: Noise levels
        heun_kwargs: Parameters for the Heun method

    Returns:
        np.ndarray: The simulated trajectories x(t)
        np.ndarray: The corresponding model predictions y(t)
        np.ndarray: The guidance weight that was computed in each step. Only not constant for 'the optimal weight
    """
    # Different initializations
    if init == 'random':
        np.random.seed(0)
        x0 = t_steps[0] * np.random.randn(n, data.shape[1])
    elif init == 'sphere':
        radius = [80]
        x0 = np.zeros((n, 2))
        split = np.array_split(np.arange(n), len(radius))
        for c, r in enumerate(radius):
            idx = np.array(list(range(c, n, len(radius))))  # Where to place the trajectories
            angles = np.linspace(0, 2 * np.pi, len(idx), endpoint=False) + 1e-2
            x0[split[c]] = r * np.vstack((np.cos(angles), np.sin(angles))).T
    else:
        x0 = init
        
    n = len(x0)
    
    # Compute trajectory
    trajects = np.empty((n, t_steps.shape[0] - 1, 2))
    preds = np.empty((n, t_steps.shape[0] - 1, 2))
    weight_hists = np.empty((n, t_steps.shape[0] - 1))
    for c, x_next in enumerate(tqdm(x0)):  # c is the trajectory "id"
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next 
            x_next, _, y_pred, cfg = heun_step(data=data,
                                               x_cur=x_cur, 
                                               t_cur=t_cur, 
                                               t_next=t_next, 
                                               i=i,
                                               c=c,
                                               **heun_kwargs)
            trajects[c, i] = x_next
            preds[c, i] = y_pred
            weight_hists[c, i] = cfg
    return trajects, preds, weight_hists


#------------------------------------------
# Plotting
def plot_traject(trajec_list, t_steps, alpha=None, scale=1, width=0.005):
    n = len(t_steps) - 2

    # Linear steps from a threshold onwards, from 1 to 0
    color_steps = np.clip(np.arange(n, 0, -1), 0, n // 2)
    c_min, c_max = color_steps.min(), color_steps.max()
    color_steps = (color_steps - c_min) / (c_max - c_min)

    # Linear steps from a threshold onwards, from 0 to 1
    alpha_steps = np.clip(np.arange(n), n // 2, n)
    c_min, c_max = alpha_steps.min(), alpha_steps.max()
    alpha_steps = (alpha_steps - c_min) / (c_max - c_min)
    for c, trajec in enumerate(trajec_list):
        plt.quiver(trajec[:-1, 0], trajec[:-1, 1], 
                   trajec[1:, 0] - trajec[:-1, 0], 
                   trajec[1:, 1] - trajec[:-1, 1], 
                   color_steps,
                   cmap='viridis',
                   scale_units='xy', 
                   angles='xy', 
                   scale=scale,
                   width=width,
                   alpha=alpha_steps if alpha is None else alpha,
                   headlength=0,
                   headaxislength=0)
        
        
def plot_endpoints(data, trajects):
    for c, trajec in enumerate(trajects):
        dist = np.power(trajec[-1] - data, 2).sum(axis=1)
        plt.scatter(trajec[-1, 0], trajec[-1, 1], s=5, c=dist.min() * 10)

        
def setup_plot(window_size, data):
    """Limit the plot to a square window of size [-window_size, window_size]"""
    plt.scatter(data[:, 0], [data[:, 1]], label='data', marker='+', s=60, color='orangered')  # firebrick
    plt.xlim(-window_size, window_size)
    plt.ylim([-window_size, window_size])
    plt.grid(alpha=0.2)
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)

    
def plot_row(n, m, i, trajects, preds, weight_hists, opt_trajects, opt_preds, opt_weight_hists,
             ylabel, titles, window_size, data, t_steps):
    """
    Args:
        n: Number of trajectories
        m: Number of trajectories to plot
        i: Row index
        trajects: Trajectories x(t), [n, n_steps, 2]
        preds: Predictions y(t), [n, n_steps, 2]
        weight_hists: Guidance weights, only for compatibility
        opt_trajects: Optimal trajectories x(t), [n, n_steps, 2)]
        opt_preds: Optimal predictions y(t), [n, n_steps, 2]
        opt_weight_hists: Optimal guidance weights, only for compatibility
        t_steps: Noise levels
    """
    rows = 4
    columns = 4
    num_steps = len(t_steps) - 1
    
    # Trajectories
    plt.subplot(rows, columns, columns*i + 1)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    if titles:
        plt.title('$x(t)$')
    
    plot_traject(trajects[::n//m], t_steps)
    setup_plot(window_size, data)
    
    traject_l2_error = (((trajects - opt_trajects)**2).sum(axis=-1) ** 0.5).mean(axis=0)  # [n, n_steps]
    plt.xlabel(f"{traject_l2_error.mean():.1e}")
     
    # Predictions
    plt.subplot(rows, columns, columns*i + 2)
    plot_traject(preds[::n//m], t_steps)
    setup_plot(window_size, data)
    if titles:
        plt.title('$y(t)$')
    pred_l2_error = (((preds - opt_preds)**2).sum(axis=-1) ** 0.5).mean()  # [n, n_steps]
    plt.xlabel(f"{pred_l2_error:.1e}")
    
    # Endpoints
    plt.subplot(rows, columns, columns*i + 3)
    plot_endpoints(data, trajects)
    setup_plot(window_size, data)
    if titles:
        plt.title('$x(0)$')
    
    total_error = (((trajects[None, :, -1] - data[:, None]) ** 2).min(axis=0).sum(axis=-1) ** 0.5).mean()
    plt.xlabel(f"{total_error:.1e}")
    
    # Trajectory error accumulation
    plt.subplot(rows, columns, columns*i + 4)
    plt.plot(np.arange(num_steps), traject_l2_error)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.grid(alpha=0.2)
    if titles:
        plt.title('L2 Error')
    
    return total_error


def full_grid(data, n, m, t_steps, guid_weight, interval, pos_kwargs, neg_kwargs, init, save_as=None):
    """
    Args:
        data: All data points
        n: Number of trajectories
        m: Number of trajectories to plot
        t_steps: Noise levels
        guid_weight: Guidance weight
        interval: Interval where the guidance weight is used
        pos_kwargs: Positive model parameters
        neg_kwargs: Negative model parameters
        init: Initialization mode
        save_as: Path to save the plot in
    """
    
    total_error = []
    returns_hist = []
    
    plt.figure(figsize=(12,12), facecolor='white')
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    window_size = 2.5
    kwargs = {'init': init, 'data': data, 'n': n, 't_steps': t_steps, 'interval': interval}
    counter = 0  # Row counter
    
    # Optimal model
    opt_returns = compute_traject(guid_weight=0,   # Only positive
                                   pos_kwargs={'delta': 0, 'cond': pos_kwargs["cond"]},  # no error
                                   neg_kwargs={'delta': 0, 'cond': False},
                                   **kwargs)
    total_error.append(plot_row(n, m, counter, *opt_returns, *opt_returns,
                                f'$\\epsilon^*(x, t)$', 
                                True, window_size, data, t_steps))
    counter += 1
    returns_hist.append(opt_returns)
    
    # WMG
    returns = compute_traject(guid_weight=guid_weight, 
                               pos_kwargs=pos_kwargs, 
                               neg_kwargs=neg_kwargs, 
                               **kwargs)
    total_error.append(plot_row(n, m, counter, *returns, *opt_returns,
                                f'$\\tilde \\epsilon(x, t)$', 
                                False, window_size, data, t_steps))
    counter += 1
    returns_hist.append(returns)
    
    # Positive trajectories
    returns = compute_traject(guid_weight=0,  # Only positive
                               pos_kwargs=pos_kwargs,
                               neg_kwargs=neg_kwargs,
                               **kwargs)
    total_error.append(plot_row(n, m, counter, *returns, *opt_returns,
                                f'$\\epsilon_{{pos}}(x, t)$', 
                                False, window_size, data, t_steps))
    counter += 1
    returns_hist.append(returns)

    # Negative trajectories
    returns = compute_traject(guid_weight=0,  # Only negative
                               pos_kwargs=neg_kwargs,
                               neg_kwargs=neg_kwargs,
                               **kwargs)
    total_error.append(plot_row(n, m, counter, *returns, *opt_returns,
                                f'$\\epsilon_{{neg}}(x, t)$', 
                                False, window_size, data, t_steps))
    counter += 1
    returns_hist.append(returns)
    
    if save_as is not None:
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
    
    plt.show()

    return returns_hist


def optimal_weights(data, n, t_steps, guid_weight, interval, pos_kwargs, neg_kwargs, init):
    """Plot the optimal guidance weight w^*(x,t)"""
    weight_hists = compute_traject(guid_weight=guid_weight,
                                   pos_kwargs=pos_kwargs,
                                   neg_kwargs=neg_kwargs,
                                   init=init,
                                   data=data,
                                   n=n,
                                   t_steps=t_steps,
                                   interval=interval)[2]

    for hist in weight_hists.mean(axis=0, keepdims=True):  # Average over trajectories
        plt.plot(hist, label=f"$\\delta_{{neg}} = {neg_kwargs['delta']}$")


#------------------------------------------
# Geometric analysis
def geom_analysis(data, t_steps, init, interval, guid_weight, steps, pos_kwargs, neg_kwargs, legend=False):
    kwargs = {'init': init, 'data': data, 'n': 1, 't_steps': t_steps, 'interval': interval}
    alpha = 0.8  # arrow brightness

    # Compute trajectory as a baseline
    x_opt, y_opt, weights_opt = compute_traject(guid_weight=guid_weight, pos_kwargs=pos_kwargs, neg_kwargs=neg_kwargs, **kwargs)
    
    for step in steps:
        # Select one trajectory and plot it
        plot_traject([x_opt[2]], t_steps, scale=0.9, width=0.008)
        x_start = x_opt[2][step-1]
        t_cur, t_next = t_steps[step], t_steps[step + 1]

        # Predictions
        y_opt = y_toy(x_start, t_cur, 2, data, **{'delta': 0, 'cond': pos_kwargs['cond']})
        y_pos = y_toy(x_start, t_cur, 2, data, **pos_kwargs)
        y_neg = y_toy(x_start, t_cur, 2, data, **neg_kwargs)

        # Optimal weight
        opt_weight = norm(y_opt - y_pos) / norm(y_pos - y_neg)
        guid_weight = guid_weight if guid_weight != 'opt_weight' else opt_weight
        y_pred = y_pos + guid_weight * (y_pos - y_neg)

        # Colors
        pos_color = 'dodgerblue'
        neg_color = 'purple'
        guid_color= 'black'
        
        
        # Positive step
        plt.quiver(x_start[0],
                   x_start[1],
                   y_pos[0] - x_start[0],
                   y_pos[1] - x_start[1],
                   color=pos_color,
                   scale_units='xy', 
                   angles='xy', 
                   scale=1,
                   label='Pos')

        # Negative step
        plt.quiver(x_start[0],
                   x_start[1],
                   y_neg[0] - x_start[0],
                   y_neg[1] - x_start[1],
                   color=neg_color,
                   scale_units='xy', 
                   angles='xy', 
                   scale=1,
                   label='Neg')

        # Guidance step
        plt.plot([y_neg[0], y_pos[0]], [y_neg[1], y_pos[1]], color='black', linestyle=':')
        plt.quiver(y_pos[0],
                   y_pos[1],
                   y_pred[0] - y_pos[0],
                   y_pred[1] - y_pos[1],
                   color=guid_color,
                   scale_units='xy', 
                   angles='xy', 
                   scale=1,
                   label='Guidance')

        if step == steps[0] and legend:
            plt.legend(prop={'size': 11})

    # Setup
    plt.scatter(data[:, 0], [data[:, 1]], label='data', marker='+', s=100*4, color='orangered', alpha=1)
    print(f'Opt weight = {opt_weight:.2f}')
