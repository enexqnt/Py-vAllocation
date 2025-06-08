from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import linalg as sla
from scipy.stats import chi2
import pandas as pd


def _cholesky_pd(mat: npt.NDArray[np.floating], jitter: float = 1e-12) -> npt.NDArray[np.floating]:
    """
    Robust Cholesky decomposition.

    If `mat` is not positive-definite, this function attempts to add a small
    multiple of the identity matrix (`jitter * I`) and retries the Cholesky
    decomposition once. This approach is discussed in Meucci's robust
    allocation framework [cite: 8], Appendix 7.2.

    Args:
        mat: The square matrix (N×N) for which to compute the Cholesky decomposition.
        jitter: The small positive constant to add to the diagonal elements
            if the matrix is not positive-definite. Defaults to 1e-12.

    Returns:
        The lower Cholesky factor of `mat` (or `mat` + `jitter` * `I`).

    Raises:
        ValueError: If `mat` is not a square matrix.
        RuntimeError: If the matrix is not positive-definite even after adding jitter.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input to _cholesky_pd must be a square matrix.")
    try:
        return sla.cholesky(mat, lower=True, check_finite=False)
    except sla.LinAlgError:
        mat_jittered = mat + jitter * np.eye(mat.shape[0])
        warnings.warn(
            "Matrix not positive-definite; added jitter for Cholesky decomposition.",
            RuntimeWarning
        )
        try:
            return sla.cholesky(mat_jittered, lower=True, check_finite=False)
        except sla.LinAlgError as exc:
            raise RuntimeError("Matrix not positive-definite after adding jitter.") from exc


def chi2_quantile(p: float, dof: int, sqrt: bool = False) -> float:
    """
    Compute the quantile of the chi-square (χ²) distribution.

    This function returns the value q such that P(X <= q) = p, where X is a
    chi-square random variable with `dof` degrees of freedom. This is used, for
    example, to determine radius factors for credibility ellipsoids as in
    Meucci's robust allocation methodology (Eqs. 6–7 in [cite: 8]).

    Args:
        p: The probability level (0 < p < 1).
        dof: The degrees of freedom for the chi-square distribution.
        sqrt: If True, returns the square root of the quantile.
            Defaults to False.

    Returns:
        The chi-square quantile Q_{χ²}(p) or sqrt(Q_{χ²}(p)) if `sqrt` is True.

    Raises:
        ValueError: If `p` is not strictly between 0 and 1.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("Probability `p` must be strictly between 0 and 1.")
    q_val = chi2.ppf(p, dof)
    return float(np.sqrt(q_val)) if sqrt else float(q_val)


@dataclass(slots=True)
class NIWParams:
    """
    Container for Normal–Inverse–Wishart (NIW) posterior parameters.

    These parameters are computed from an NIW prior and sample statistics
    following the update rules in Meucci's framework (Eqs. 11–14 in [cite: 8]).

    Attributes:
        T1: Posterior pseudo-count for the mean (scalar integer)[cite: 8].
        mu1: Posterior mean vector as a NumPy array or pandas.Series[cite: 8].
        nu1: Posterior pseudo-count for the covariance (scalar integer)[cite: 8].
        sigma1: Posterior scale matrix for the covariance (NumPy array or pandas.DataFrame)[cite: 8].
    """
    T1: int
    mu1: Union[npt.NDArray[np.floating], "pd.Series[np.floating]"]
    nu1: int
    sigma1: Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"]


class NIWPosterior:
    """
    Computes and manages Normal–Inverse–Wishart (NIW) posterior parameters.

    This class implements the Bayesian update rules for an NIW distribution,
    assuming normally distributed returns and an NIW prior. The updates follow
    Eqs. 11–14 in Meucci's framework [cite: 8]. It also provides methods to
    calculate classical-equivalent estimators and credibility-related factors
    used in robust Bayesian asset allocation[cite: 8].

    The NIW distribution is a conjugate prior for a multivariate normal
    distribution with unknown mean and covariance matrix[cite: 8].

    How to Use:
    1.  Initialize the `NIWPosterior` object with prior parameters:
        * `prior_mu` ($\mu_0$): Your initial estimate for the mean vector, can be
          a NumPy array or pandas.Series.
        * `prior_sigma` ($\Sigma_0$): Your initial estimate for the scale matrix of
          the covariance, can be a NumPy array or pandas.DataFrame.
        * `t0` ($T_0$): Your confidence in `prior_mu`, expressed as a pseudo-count of observations[cite: 8].
        * `nu0` ($\nu_0$): Your confidence in `prior_sigma`, expressed as a pseudo-count of observations[cite: 8].
    2.  Call the `update()` method with sample statistics derived from observed data:
        * `sample_mu` ($\hat{\mu}$): The mean vector calculated from your sample data,
          can be a NumPy array or pandas.Series.
        * `sample_sigma` ($\hat{\Sigma}$): The covariance matrix calculated from your
          sample data, can be a NumPy array or pandas.DataFrame.
        * `n_obs` ($T$): The number of observations in your sample data[cite: 8].
    3.  The `update()` method returns an `NIWParams` object containing the posterior
        parameters ($T_1, \mu_1, \nu_1, \Sigma_1$). If inputs were pandas, then
        $\mu_1$ is a pandas.Series and $\Sigma_1$ is a pandas.DataFrame.
    4.  Use accessor methods like `get_posterior()`, `get_mu_ce()`, `get_S_mu()`,
        `get_sigma_ce()`, `cred_radius_mu()`, and `cred_radius_sigma_factor()` to
        retrieve various posterior quantities. These also respect pandas formats.

    Attributes:
        prior_mu (pd.Series or np.ndarray): The prior mean vector (μ₀) if pandas, with index.
        prior_sigma (pd.DataFrame or np.ndarray): The prior scale matrix (Σ₀) if pandas,
            with index and columns.
        t0 (int): The prior pseudo-count for the mean (T₀).
        nu0 (int): The prior pseudo-count for the covariance (ν₀).
        N (int): The number of assets.
        _asset_index (pd.Index or None): Index of assets if prior_mu was pandas.Series.
        _posterior (Optional[NIWParams]): Stores the computed posterior parameters.
    """

    def __init__(
        self,
        prior_mu: Union[npt.NDArray[np.floating], "pd.Series[np.floating]"],
        prior_sigma: Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"],
        t0: int,
        nu0: int,
    ) -> None:
        """
        Initializes the NIWPosterior object with prior parameters.

        Args:
            prior_mu: 1D array (length N) of prior means (μ₀), or pandas.Series.
            prior_sigma: 2D array (N×N) of the prior scale matrix (Σ₀), or pandas.DataFrame.
            t0: Prior pseudo-count for the mean (T₀). Must be > 0[cite: 8].
            nu0: Prior pseudo-count for the covariance (ν₀). Must be ≥ 0[cite: 8].

        Raises:
            ValueError: If input parameters have inconsistent shapes or invalid values.
        """
        # Detect pandas inputs
        self._pandas = False
        self._asset_index: Optional[pd.Index] = None

        if isinstance(prior_mu, pd.Series):
            self._pandas = True
            self._asset_index = prior_mu.index.copy()
            self.prior_mu: npt.NDArray[np.floating] = prior_mu.values.astype(float)
        else:
            self.prior_mu = np.asarray(prior_mu, dtype=float)

        if self.prior_mu.ndim != 1:
            raise ValueError("`prior_mu` must be a 1D array or pandas.Series.")
        self.N: int = self.prior_mu.size
        if self.N == 0:
            raise ValueError("`prior_mu` cannot be empty; N must be > 0.")

        if isinstance(prior_sigma, pd.DataFrame):
            if self._asset_index is None:
                # If prior_mu was not pandas but prior_sigma is, infer index
                self._pandas = True
                self._asset_index = prior_sigma.index.copy()
            else:
                # Ensure indices match
                if not prior_sigma.index.equals(self._asset_index) or not prior_sigma.columns.equals(self._asset_index):
                    raise ValueError("Index/columns of prior_sigma must match index of prior_mu.")
            self.prior_sigma: npt.NDArray[np.floating] = prior_sigma.values.astype(float)
        else:
            self.prior_sigma = np.asarray(prior_sigma, dtype=float)

        if self.prior_sigma.ndim != 2 or self.prior_sigma.shape != (self.N, self.N):
            raise ValueError(f"`prior_sigma` must be a square matrix of shape ({self.N}, {self.N}), or a pandas.DataFrame with matching index/columns.")

        if t0 <= 0:
            raise ValueError("`t0` (prior pseudo-count for mean) must be strictly positive.")
        if nu0 < 0:
            raise ValueError("`nu0` (prior pseudo-count for covariance) must be non-negative.")

        _ = _cholesky_pd(self.prior_sigma)  # Ensure prior_sigma is PD or near-PD

        self.t0: int = int(t0)
        self.nu0: int = int(nu0)
        self._posterior: Optional[NIWParams] = None

    def update(
        self,
        sample_mu: Union[npt.NDArray[np.floating], "pd.Series[np.floating]"],
        sample_sigma: Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"],
        n_obs: int,
    ) -> NIWParams:
        """
        Updates the posterior parameters using sample statistics.

        This method implements Eqs. 11–14 from Meucci's framework [cite: 8]:
          * $T_1 = T_0 + T$ [cite: 8]
          * $\mu_1 = (T_0 \mu_0 + T \hat{\mu}) / T_1$ [cite: 8]
          * $\nu_1 = \nu_0 + T$ [cite: 8]
          * $\Sigma_1 = [\nu_0 \Sigma_0 + T \hat{\Sigma} + (\mu_0 - \hat{\mu})(\mu_0 - \hat{\mu})^T / (1/T + 1/T_0)] / \nu_1$ [cite: 8]

        Args:
            sample_mu: 1D array (length N) of sample means ($\hat{\mu}$) or pandas.Series.
            sample_sigma: 2D array (N×N) of sample covariance matrix ($\hat{\Sigma}$) or pandas.DataFrame.
            n_obs: Number of observations in the sample (T)[cite: 8]. Must be > 0.

        Returns:
            A NIWParams instance containing the updated posterior parameters (T₁, μ₁, ν₁, Σ₁).
            If inputs were pandas, then μ₁ is returned as a pandas.Series and Σ₁ as a pandas.DataFrame.
        Raises:
            ValueError: If sample statistics have inconsistent shapes or `n_obs` is invalid.
        """
        # Convert pandas inputs if present
        if isinstance(sample_mu, pd.Series):
            smu = sample_mu.values.astype(float)
        else:
            smu = np.asarray(sample_mu, dtype=float)

        if smu.ndim != 1 or smu.shape[0] != self.N:
            raise ValueError(f"`sample_mu` must be a 1D array or pandas.Series of length {self.N}.")

        if n_obs <= 0:
            raise ValueError("`n_obs` (number of observations) must be strictly positive.")

        if isinstance(sample_sigma, pd.DataFrame):
            ssigma = sample_sigma.values.astype(float)
        else:
            ssigma = np.asarray(sample_sigma, dtype=float)

        if ssigma.ndim != 2 or ssigma.shape != (self.N, self.N):
            raise ValueError(f"`sample_sigma` must be a square matrix of shape ({self.N}, {self.N}), or pandas.DataFrame with matching index/columns.")

        _ = _cholesky_pd(ssigma)  # Ensure sample_sigma is PD or near-PD

        # Compute posterior scalars
        T1 = self.t0 + n_obs     # Eq. (11) [cite: 8]
        nu1 = self.nu0 + n_obs   # Eq. (13) [cite: 8]
        mu1_array = (self.t0 * self.prior_mu + n_obs * smu) / T1  # Eq. (12) [cite: 8]

        cross_term_weight_denominator = (1.0 / n_obs + 1.0 / self.t0)
        diff_mu = self.prior_mu - smu
        outer_prod_diff_mu = np.outer(diff_mu, diff_mu)
        cross_term_weighted = outer_prod_diff_mu / cross_term_weight_denominator

        sigma1_numerator = (self.nu0 * self.prior_sigma
                            + n_obs * ssigma
                            + cross_term_weighted)
        if nu1 <= 0:
            raise ValueError("Posterior degrees of freedom ν₁ must be positive.")
        sigma1_array = sigma1_numerator / nu1  # Eq. (14) [cite: 8]

        _ = _cholesky_pd(sigma1_array)  # Ensure Σ₁ is PD or near-PD

        # Wrap into pandas if appropriate
        if self._pandas and self._asset_index is not None:
            mu1: Union[npt.NDArray[np.floating], "pd.Series[np.floating]"] = pd.Series(
                mu1_array, index=self._asset_index
            )
            sigma1: Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"] = pd.DataFrame(
                sigma1_array, index=self._asset_index, columns=self._asset_index
            )
        else:
            mu1 = mu1_array
            sigma1 = sigma1_array

        self._posterior = NIWParams(T1=T1, mu1=mu1, nu1=nu1, sigma1=sigma1)
        return self._posterior

    def get_posterior(self) -> Optional[NIWParams]:
        """
        Retrieves the computed posterior parameters.

        Returns:
            A NIWParams instance containing (T₁, μ₁, ν₁, Σ₁), or None if `update()` has not yet been called.
        """
        return self._posterior

    def get_mu_ce(self) -> Union[npt.NDArray[np.floating], "pd.Series[np.floating]"]:
        """
        Computes the classical-equivalent estimator for the mean ($\hat{\mu}_{ce}$).

        As per Meucci's framework (Eq. 15 in [cite: 8]), $\hat{\mu}_{ce} = \mu_1$.
        Returns as a pandas.Series if prior_mu was provided as pandas.Series.

        Returns:
            The posterior mean vector $\mu_1$ as a NumPy array or pandas.Series.

        Raises:
            RuntimeError: If posterior parameters have not been computed via `update()`.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        mu1 = self._posterior.mu1
        # If stored as numpy but we want to return pandas, wrap here:
        if self._pandas and not isinstance(mu1, pd.Series) and self._asset_index is not None:
            return pd.Series(mu1, index=self._asset_index)
        return mu1

    def get_S_mu(self) -> Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"]:
        """
        Computes the scatter matrix $S_{\mu}$ for the marginal posterior distribution of $\mu$.

        As per Eq. 16 in Meucci's framework[cite: 8],
        $S_{\mu} = (1 / T_1) * (\nu_1 / (\nu_1 - 2)) * \Sigma_1$.
        This quantity is used in defining the location-dispersion ellipsoid for $\mu$[cite: 8].
        Requires $\nu_1 > 2$. Returns as a pandas.DataFrame if prior_sigma was pandas.DataFrame.

        Returns:
            The scatter matrix $S_{\mu}$ as a NumPy array or pandas.DataFrame.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If $\nu_1 \le 2$, as $S_{\mu}$ is undefined or problematic.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        if self._posterior.nu1 <= 2:
            raise ValueError("Posterior degrees of freedom ν₁ must be greater than 2 to compute S_μ.")
        factor = self._posterior.nu1 / (self._posterior.T1 * (self._posterior.nu1 - 2.0))
        S_mu_array = (
            self._posterior.sigma1.values
            if isinstance(self._posterior.sigma1, pd.DataFrame)
            else self._posterior.sigma1
        )
        if self._pandas and self._asset_index is not None:
            return pd.DataFrame(S_mu_array, index=self._asset_index, columns=self._asset_index)
        return S_mu_array

    def get_sigma_ce(self) -> Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"]:
        """
        Computes the classical-equivalent estimator for the covariance matrix ($\hat{\Sigma}_{ce}$).

        As per Eq. 17 in Meucci's framework[cite: 8],
        $\hat{\Sigma}_{ce} = (\nu_1 / (\nu_1 + N + 1)) * \Sigma_1$.
        Returns as a pandas.DataFrame if prior_sigma was provided as pandas.DataFrame.

        Returns:
            The classical-equivalent estimator $\hat{\Sigma}_{ce}$ as a NumPy array or pandas.DataFrame.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If $\nu_1 + N + 1 = 0$, though unlikely with valid inputs.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        denom = self._posterior.nu1 + self.N + 1.0
        if denom == 0:
            raise ValueError("Denominator (ν₁ + N + 1) for Σ_ce is zero.")
        factor = self._posterior.nu1 / denom
        sigma1_array = (
            self._posterior.sigma1.values
            if isinstance(self._posterior.sigma1, pd.DataFrame)
            else self._posterior.sigma1
        )
        sigma_ce_array = factor * sigma1_array
        if self._pandas and self._asset_index is not None:
            return pd.DataFrame(sigma_ce_array, index=self._asset_index, columns=self._asset_index)
        return sigma_ce_array

    def cred_radius_mu(self, p_mu: float) -> float:
        """
        Computes the credibility factor γ_μ for the mean's uncertainty ellipsoid.

        As per Eq. 20 in Meucci's framework[cite: 8],
        γ_μ = sqrt[(q_μ^2 / T₁) * (ν₁ / (ν₁ − 2))]
        with $q_μ^2 = Q_{χ²_N}(p_mu)$ as in Meucci[cite: 8].

        Args:
            p_mu: Confidence level for μ (0 < p_mu < 1).

        Returns:
            The credibility factor γ_μ as a float.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If ν₁ ≤ 2 or p_mu is not in (0,1).
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        if self._posterior.nu1 <= 2:
            raise ValueError("Posterior ν₁ must be greater than 2 for γ_μ calculation.")
        if not (0.0 < p_mu < 1.0):
            raise ValueError("Confidence level p_mu must be between 0 and 1 (exclusive).")
        q_mu_squared = chi2_quantile(p_mu, self.N, sqrt=False)
        term_T1 = self._posterior.T1
        term_nu1 = self._posterior.nu1
        gamma_mu = np.sqrt((q_mu_squared / term_T1) * (term_nu1 / (term_nu1 - 2.0)))
        return gamma_mu

    def cred_radius_sigma_factor(self, p_sigma: float) -> float:
        """
        Computes the credibility factor C_Σ related to Σ for robust allocation.

        As per Eq. 47 in Meucci's framework[cite: 8],
        C_Σ = [ν₁ / (ν₁ + N + 1)] + sqrt[2 ν₁² q_Σ² / (ν₁ + N + 1)³],
        where $q_Σ² = Q_{χ²_{dof}}(p_Σ)$ with $dof = N(N+1)/2$.

        Args:
            p_sigma: Confidence level for Σ (0 < p_sigma < 1).

        Returns:
            The credibility factor C_Σ as a float.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If p_sigma not in (0,1) or invalid internal state.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        if not (0.0 < p_sigma < 1.0):
            raise ValueError("Confidence level p_sigma must be between 0 and 1 (exclusive).")

        nu1 = self._posterior.nu1
        denom_cubed_base = nu1 + self.N + 1.0
        if denom_cubed_base <= 0:
            raise ValueError("Term (ν₁ + N + 1) must be positive for C_Σ calculation.")

        dof = self.N * (self.N + 1) // 2
        q_sigma_squared_val = chi2_quantile(p_sigma, dof, sqrt=False)

        term1 = nu1 / denom_cubed_base
        numerator_term2 = 2.0 * nu1**2 * q_sigma_squared_val
        denominator_term2 = denom_cubed_base**3
        if denominator_term2 == 0:
            raise ValueError("Denominator for sqrt term in C_Σ is zero.")
        term2_arg = numerator_term2 / denominator_term2
        if term2_arg < 0:
            warnings.warn(
                f"Argument for sqrt in C_Σ calculation is negative ({term2_arg:?}); result may be NaN.",
                RuntimeWarning
            )
        term2 = np.sqrt(term2_arg)
        C_sigma = term1 + term2
        return C_sigma
