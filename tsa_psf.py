import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker



def visualize_psfs(psfs, ps=0, ncols=10, title='PSFs', cmap='gray', origin='lower', interpolation='none'):
    """
    Visualize a set of PSFs by plotting them.

    Parameters
    ----------
    psfs : array-like, shape (num_psfs, psf_size, psf_size)
        Images of PSFs.
    ps : int
        Optional padding crop (removes ps pixels around each edge).
    ncols : int
        Number of columns in PSF visualization grid.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap.
    origin : str
        Matplotlib origin parameter for imshow method.
    interpolation : str
        Matplotlib interpolation parameter for imshow method.
    """

    # Get PSF data dimensions
    num_psfs = psfs.shape[0]
    psf_size = psfs.shape[1]
    
    # Grid layout
    ncols = ncols
    nrows = int(np.ceil(num_psfs / ncols))

    # Define figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        if i < num_psfs:
            axes[i].imshow(psfs[i, ps:-ps, ps:-ps], cmap=cmap, origin=origin, interpolation=interpolation)
            axes[i].set_xticks([])  # hide ticks & spines but keep the axis visible so titles/labels render
            axes[i].set_yticks([])
            for s in axes[i].spines.values():
                s.set_visible(False)
            axes[i].set_title(f"PSF {i+1}", fontsize=15, pad=4) 
        else:
            axes[i].set_visible(False)  # hide empty subplots

    fig.suptitle(title, fontsize=20, y=1.01)  # bump up so it clears the grid
    plt.show()

    return fig, axes
    


def compare_psfs(psfs_0, psfs_1, ps=0, n_show=20,
                 titles=['Original','Reconstruction','Error'],
                 cmap='gray', origin='lower', interpolation='none'):
    """
    Compare two sets of PSFs by plotting them, and their difference, side by side.

    Parameters
    ----------
    psfs_0 : array-like, shape (num_psfs, psf_size, psf_size)
        First set of PSFs, e.g., Ground-truth/original PSF images.
    psfs_1 : array-like, shape (num_psfs, psf_size, psf_size)
        Second set of PSFs, e.g., Reconstructed PSF images.
    ps : int
        Optional padding crop (removes ps pixels around each edge).
    n_show : int
        Number of examples to display.
    titles: list of str (len=3)
        Titles for plots.
    cmap : str
        Matplotlib colormap.
    origin : str
        Matplotlib origin parameter for imshow method.
    interpolation : str
        Matplotlib interpolation parameter for imshow method.
    """
    
    # Formatter for 3 decimal places
    formatter = mticker.FormatStrFormatter('%.3f')

    # Update number of examples to display
    N = min(len(psfs_0), len(psfs_1), n_show)

    for i in range(N):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # First set of PSFs
        im0 = axes[0].imshow(psfs_0[i, ps:-ps or None, ps:-ps or None],
                             cmap=cmap, origin=origin, interpolation=interpolation)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        for s in axes[0].spines.values():
            s.set_visible(False)
        axes[0].set_title(f"{titles[0]} {i}")
        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, format=formatter)

        # Second set of PSFs
        im1 = axes[1].imshow(psfs_1[i, ps:-ps or None, ps:-ps or None],
                             cmap=cmap, origin=origin, interpolation=interpolation)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        for s in axes[1].spines.values():
            s.set_visible(False)
        axes[1].set_title(f"{titles[1]} {i}")
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, format=formatter)

        # Difference between sets of PSFs
        error = psfs_0[i] - psfs_1[i]
        im2 = axes[2].imshow(error[ps:-ps or None, ps:-ps or None],
                             cmap=cmap, origin=origin, interpolation=interpolation)
        axes[2].set_xticks([]); axes[2].set_yticks([])
        for s in axes[2].spines.values():
            s.set_visible(False)
        axes[2].set_title(f"{titles[2]} {i}")
        cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, format=formatter)

        plt.show()
        


def generate_multivariate_ar(T=100, D=20, lags=[5, 10], coefs=[0.5, 0.3], noise_std=1.0, seed=None):
    """
    Generate synthetic multivariate time series with specified autocorrelations.

    Parameters
    ----------
    T : int
        Number of time points.
    D : int
        Number of dimensions (variables).
    lags : list of int
        Time lags where autocorrelations occur.
    coefs : list of float
        Coefficients for each lag (same length as lags).
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray, shape (T, D)
        Generated time series data.
    """

    # Set default seed for random number generation
    rng = np.random.default_rng(seed)
    data = np.zeros((T, D))

    # Start with random initial conditions
    data[:max(lags)] = rng.normal(scale=noise_std, size=(max(lags), D))

    # Generate AR process
    for t in range(max(lags), T):
        for lag, coef in zip(lags, coefs):
            data[t] += coef * data[t - lag]
        data[t] += rng.normal(scale=noise_std, size=D)

    return data
