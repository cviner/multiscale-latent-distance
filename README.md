# multiscale-latent-distance

## Installation

```
pip install "msld @ git+https://github.com/cviner/multiscale-latent-distance@v1.0.0"
```

## Usage

```python
import msld

# counts: array of shape (S, n, n), symmetric in each channel; the diagonal is ignored
# widths: the S variances sigma_s^2
# x0: starting coordinates, shape (n, d)
(x, a, b), record = msld.fit(counts, widths, x0)
eta = msld.log_mean(x, a, b, widths)  # fitted log means, shape (S, n, n)
```

`msld.fit_multistart` takes several starts and keeps the lowest objective. `msld.scaled_objective` returns the objective and gradient for calling `scipy.optimize.minimize` directly.

## Model P

Each item $i$ has coordinates $x_i$ and an effect $a_i$, and each channel $s$ has an intercept $b_s$ and a fixed width $\sigma_s$. For $i\neq j$, the expected observation in channel $s$ is

$$\lambda^{(s)}_{ij}=\exp\left(b_s+a_i+a_j-\frac{\lVert x_i-x_j\rVert^2}{2\sigma_s^2}\right).$$

The fit minimizes the Poisson objective

$$Q=\sum_s\sum_{i\neq j}\left[\lambda^{(s)}_{ij}-C^{(s)}_{ij}\log\lambda^{(s)}_{ij}\right]$$

over all pairs, including those with no observation, so time and memory grow with the square of the number of items.

## Licence

GNU General Public License, version 3 (see `LICENSE`).
