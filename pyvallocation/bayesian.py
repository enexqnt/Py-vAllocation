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
    r"""Computes a Cholesky decomposition that is robust to numerical errors.

    The theoretical framework for robust Bayesian allocation often requires
    operations like spectral or Cholesky decompositions of covariance matrices,
    which must be positive-definite (PD). However, matrices estimated
    from real-world data may fail to be perfectly PD due to floating-point
    inaccuracies or estimation errors.

    This function attempts to compute the Cholesky decomposition of `mat`. If
    `mat` is not PD, it adds a small multiple of the identity matrix
    (`jitter` * I) to the matrix to make it PD and then retries. This is a
    standard numerical stabilization technique.

    Args:
        mat: The square matrix (N x N) for which to compute the Cholesky
            decomposition.
        jitter: A small positive constant to add to the diagonal elements if
            the matrix is not positive-definite. Defaults to 1e-12.

    Returns:
        The lower Cholesky factor of `mat` (or `mat` + `jitter` * I).

    Raises:
        ValueError: If `mat` is not a square matrix.
        RuntimeError: If the matrix is not positive-definite even after
            adding jitter.
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
    r"""Computes the quantile of the chi-square (χ²) distribution.

    This function is used to determine the size of the uncertainty ellipsoids
    for the market parameters (mean and covariance). The size is
    determined by a radius factor, :math:`q`, which is set according to a
    quantile of the chi-square distribution.

    For the mean vector :math:`\mu`, under the assumption that its posterior
    distribution is normal, the squared Mahalanobis distance is chi-square
    distributed. The radius factor squared, :math:`q_\mu^2`, is set using a
    quantile of the :math:`\chi^2_N` distribution.

    .. math::
        q_\mu^2 = Q_{\chi_N^2}(p_\mu)

    For the covariance matrix :math:`\Sigma`, a similar approach is used based on a
    heuristic argument that the Mahalanobis distance behaves like a
    :math:`\chi^2` distribution with :math:`N(N+1)/2` degrees of freedom
    .

    .. math::
        q_\Sigma^2 = Q_{\chi_{N(N+1)/2}^2}(p_\Sigma)

    Args:
        p: The probability level (0 < p < 1) for the quantile.
        dof: The degrees of freedom for the chi-square distribution.
        sqrt: If True, returns the square root of the quantile. Defaults to False.

    Returns:
        The chi-square quantile :math:`Q_{\chi²}(p)` or :math:`\sqrt{Q_{\chi²}(p)}`
        if `sqrt` is True.

    Raises:
        ValueError: If `p` is not strictly between 0 and 1.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("Probability `p` must be strictly between 0 and 1.")
    q_val = chi2.ppf(p, dof)
    return float(np.sqrt(q_val)) if sqrt else float(q_val)


@dataclass()
class NIWParams:
    r"""A container for the parameters of a Normal-Inverse-Wishart (NIW) posterior distribution. :no-inheritance:

    These parameters are the result of a Bayesian update, combining an
    NIW prior with market data, as detailed in Meucci (2005). The
    formulas for these posterior parameters are given in Eqs. (11)-(14).
    """
    T1: int
    #: The posterior pseudo-observations for the mean (:math:`T_1`),
    #: representing the updated confidence in the mean estimate.
    
    mu1: Union[npt.NDArray[np.floating], "pd.Series[np.floating]"]
    #: The posterior mean vector (:math:`\mu_1`), which is the updated
    #: estimate of the expected returns.
    
    nu1: int
    #: The posterior degrees of freedom for the covariance
    #: (:math:`\nu_1`), representing the updated confidence in the
    #: covariance estimate.
    
    sigma1: Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"]
    #: The posterior scale matrix for the covariance (:math:`\Sigma_1`),
    #: which is the updated scale matrix of the Inverse-Wishart
    #: distribution.

class NIWPosterior:
    r"""Computes and manages Normal-Inverse-Wishart (NIW) posterior parameters.

    This class implements the Bayesian update rules for an NIW distribution,
    which is the conjugate prior for a multivariate normal likelihood with
    unknown mean and covariance. The methodology follows Section 3
    of Meucci (2005). It provides methods to calculate posterior
    parameters, classical-equivalent estimators, and factors used in robust
    Bayesian asset allocation.

    The model assumes that asset returns are independently and identically
    distributed according to a normal distribution. The investor's
    prior knowledge is modeled as an NIW distribution.

    How to Use:

    1.  Initialize the :class:`NIWPosterior` object with prior parameters:

        * ``prior_mu`` (:math:`\mu_0`): The prior estimate for the mean vector.
        * ``prior_sigma`` (:math:`\Sigma_0`): The prior scale matrix for the covariance.
        * ``t0`` (:math:`T_0`): The confidence in ``prior_mu``, expressed as a
            pseudo-count of observations.
        * ``nu0`` (:math:`\nu_0`): The confidence in ``prior_sigma``, expressed as a
            pseudo-count of observations.

    2.  Call the :meth:`update` method with sample statistics from observed data:

        * ``sample_mu`` (:math:`\hat{\mu}`): The mean vector from the data.
        * ``sample_sigma`` (:math:`\hat{\Sigma}`): The covariance matrix from the data.
        * ``n_obs`` (:math:`T`): The number of observations in the data sample.

    3.  The :meth:`update` method returns an :class:`NIWParams` object with the
        posterior parameters (:math:`T_1, \mu_1, \nu_1, \Sigma_1`), which are
        a blend of the prior and the market data.

    4.  Use accessor methods like :meth:`get_mu_ce`, :meth:`get_S_mu`, etc., to
        retrieve various quantities derived from the posterior distribution.

    Attributes:
        prior_mu: The prior mean vector (:math:`\mu_0`).
        prior_sigma: The prior scale matrix (:math:`\Sigma_0`).
        t0: The prior pseudo-count for the mean (:math:`T_0`).
        nu0: The prior pseudo-count for the covariance (:math:`\nu_0`).
        N: The number of assets.
        _asset_index: Stores pandas.Index if pandas objects are used.
        _posterior: Stores the computed posterior parameters.
    """

    def __init__(
        self,
        prior_mu: Union[npt.NDArray[np.floating], "pd.Series[np.floating]"],
        prior_sigma: Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"],
        t0: int,
        nu0: int,
    ) -> None:
        r"""Initializes the NIWPosterior object with prior parameters.

        The prior parameters :math:`(\mu_0, \Sigma_0)` represent the investor's
        experience, while :math:`(T_0, \nu_0)` represent their confidence in
        that experience.

        Args:
            prior_mu: 1D array (length N) of prior means (:math:`\mu_0`).
            prior_sigma: 2D array (N x N) of the prior scale matrix (:math:`\Sigma_0`).
            t0: Prior pseudo-count for the mean (:math:`T_0`). Must be > 0.
            nu0: Prior pseudo-count for the covariance (:math:`\nu_0`). Must be >= 0.

        Raises:
            ValueError: If input parameters have inconsistent shapes or
                invalid values.
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
        r"""Updates the posterior parameters using sample statistics.

        This method implements the Bayesian update rules for the NIW parameters
        as given by Eqs. (11)–(14) in Meucci (2005).

        .. math::
            \begin{align*}
            T_1 &= T_0 + T \\
            \mu_1 &= \frac{T_0\mu_0 + T\hat{\mu}}{T_1} \\
            \nu_1 &= \nu_0 + T \\
            \Sigma_1 &= \frac{1}{\nu_1} \left[ \nu_0\Sigma_0 + T\hat{\Sigma} + \frac{(\mu_0 - \hat{\mu})(\mu_0 - \hat{\mu})'}{\frac{1}{T} + \frac{1}{T_0}} \right]
            \end{align*}

        The resulting posterior parameters blend the investor's prior with
        information from the market, with the balance determined by the
        relative confidence levels (:math:`T_0, \nu_0`) versus the amount of
        data (:math:`T`).

        Args:
            sample_mu: 1D array (length N) of sample means (:math:`\hat{\mu}`).
            sample_sigma: 2D array (N x N) of the sample covariance matrix
                (:math:`\hat{\Sigma}`).
            n_obs: The number of observations in the sample (:math:`T`).

        Returns:
            A :class:`NIWParams` instance with the updated posterior parameters.
            If pandas objects were used in initialization, returns :math:`\mu_1`
            as a Series and :math:`\Sigma_1` as a DataFrame.

        Raises:
            ValueError: If sample statistics have inconsistent shapes or ``n_obs``
                is not a positive integer.
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
        T1 = self.t0 + n_obs
        nu1 = self.nu0 + n_obs
        mu1_array = (self.t0 * self.prior_mu + n_obs * smu) / T1

        cross_term_weight_denominator = (1.0 / n_obs + 1.0 / self.t0)
        diff_mu = self.prior_mu - smu
        outer_prod_diff_mu = np.outer(diff_mu, diff_mu)
        cross_term_weighted = outer_prod_diff_mu / cross_term_weight_denominator

        sigma1_numerator = (self.nu0 * self.prior_sigma
                            + n_obs * ssigma
                            + cross_term_weighted)
        if nu1 <= 0:
            raise ValueError("Posterior degrees of freedom ν₁ must be positive.")
        sigma1_array = sigma1_numerator / nu1

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
        r"""Retrieves the computed posterior parameters.

        Returns:
            A :class:`.NIWParams` instance containing the posterior parameters
            (:math:`T_1`, :math:`\mu_1`, :math:`\nu_1`, :math:`\Sigma_1`), or ``None``
            if :meth:`~.NIWPosterior.update` has not been called.

        """
        return self._posterior

    def get_mu_ce(self) -> Union[npt.NDArray[np.floating], "pd.Series[np.floating]"]:
        r"""Computes the classical-equivalent estimator for the mean, :math:`\hat{\mu}_{ce}`.

        For the NIW model, this estimator is the posterior mean :math:`\mu_1`,
        as defined in Eq. (15).

        .. math::
            \hat{\mu}_{ce} = \mu_1

        Returns:
            The posterior mean vector :math:`\mu_1` as a NumPy array or pandas
            Series.

        Raises:
            RuntimeError: If posterior parameters have not been computed via
                :meth:`update`.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        mu1 = self._posterior.mu1
        # If stored as numpy but we want to return pandas, wrap here:
        if self._pandas and not isinstance(mu1, pd.Series) and self._asset_index is not None:
            return pd.Series(mu1, index=self._asset_index)
        return mu1

    def get_S_mu(self) -> Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"]:
        r"""Computes the scatter matrix :math:`S_{\mu}` for the posterior of :math:`\mu`.

        This matrix describes the dispersion of the marginal posterior
        distribution of :math:`\mu` and is used to define the location-dispersion
        ellipsoid for robust optimization. It is defined in Eq. (16):

        .. math::
            S_{\mu} = \frac{1}{T_1} \frac{\nu_1}{\nu_1 - 2} \Sigma_1

        This computation requires the posterior degrees of freedom :math:`\nu_1 > 2`.

        Returns:
            The scatter matrix :math:`S_{\mu}` as a NumPy array or pandas DataFrame.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If :math:`\nu_1 \le 2`, as the scatter matrix is not
                defined.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        if self._posterior.nu1 <= 2:
            raise ValueError("Posterior degrees of freedom ν₁ must be greater than 2 to compute S_μ.")
        factor = self._posterior.nu1 / (self._posterior.T1 * (self._posterior.nu1 - 2.0))
        # Ensure underlying data is array for multiplication
        sigma1_array = self._posterior.sigma1
        if isinstance(sigma1_array, pd.DataFrame):
            sigma1_array = sigma1_array.values

        S_mu_array = factor * sigma1_array

        if self._pandas and self._asset_index is not None:
            return pd.DataFrame(S_mu_array, index=self._asset_index, columns=self._asset_index)
        return S_mu_array


    def get_sigma_ce(self) -> Union[npt.NDArray[np.floating], "pd.DataFrame[np.floating]"]:
        r"""Computes the classical-equivalent estimator for the covariance, :math:`\hat{\Sigma}_{ce}`.

        This estimator is a shrunk version of the posterior scale matrix
        :math:`\Sigma_1`, as defined in Eq. (17). It serves as the center of
        the uncertainty ellipsoid for :math:`\Sigma`.

        .. math::
            \hat{\Sigma}_{ce} = \frac{\nu_1}{\nu_1 + N + 1} \Sigma_1

        Returns:
            The classical-equivalent estimator :math:`\hat{\Sigma}_{ce}` as a NumPy
            array or pandas DataFrame.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If :math:`\nu_1 + N + 1 = 0`, which is highly unlikely
                with valid inputs.
        """
        if self._posterior is None:
            raise RuntimeError("Posterior parameters not computed. Call `update()` first.")
        denom = self._posterior.nu1 + self.N + 1.0
        if denom == 0:
            raise ValueError("Denominator (ν₁ + N + 1) for Σ_ce is zero.")
        factor = self._posterior.nu1 / denom

        sigma1_array = self._posterior.sigma1
        if isinstance(sigma1_array, pd.DataFrame):
            sigma1_array = sigma1_array.values

        sigma_ce_array = factor * sigma1_array
        if self._pandas and self._asset_index is not None:
            return pd.DataFrame(sigma_ce_array, index=self._asset_index, columns=self._asset_index)
        return sigma_ce_array

    def cred_radius_mu(self, p_mu: float) -> float:
        r"""Computes the credibility factor :math:`\gamma_\mu` for the mean's uncertainty.

        This factor, :math:`\gamma_\mu`, appears in the simplified robust
        mean-variance optimization problem (Eq. 19). It scales the
        portfolio's posterior standard deviation to penalize for estimation
        risk in the mean vector. Its formula is given by Eq. (20):

        .. math::
            \gamma_\mu = \sqrt{ \frac{q_\mu^2}{T_1} \frac{\nu_1}{\nu_1 - 2} }

        where :math:`q_\mu^2 = Q_{\chi^2_N}(p_{mu})` is the squared radius factor
        from the chi-square distribution.

        Args:
            p_mu: The confidence level for :math:`\mu` (0 < p_mu < 1), which
                reflects aversion to estimation risk.

        Returns:
            The credibility factor :math:`\gamma_\mu`.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If :math:`\nu_1 \le 2` or `p_mu` is not in (0,1).
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
        r"""Computes the scaling factor for the worst-case portfolio variance.

        In the robust framework, the maximum possible variance of a portfolio
        within the uncertainty ellipsoid for :math:`\Sigma` is not simply
        :math:`w'\Sigma_1 w`, but a scaled version of it. This method computes
        that scaling factor, which we can call :math:`C_\Sigma`. The derivation is
        shown in Appendix 7.2, and the final result is presented in the
        maximization step in Eq. (47):

        .. math::
            \max_{\Sigma \in \Theta_\Sigma} w'\Sigma w = \underbrace{ \left[ \frac{\nu_1}{\nu_1 + N + 1} + \sqrt{\frac{2\nu_1^2 q_\Sigma^2}{(\nu_1 + N + 1)^3}} \right] }_{C_\Sigma} (w'\Sigma_1 w)

        where :math:`q_\Sigma^2 = Q_{\chi^2_{dof}}(p_\Sigma)` with
        :math:`dof = N(N+1)/2`.

        Args:
            p_sigma: The confidence level for :math:`\Sigma` (0 < p_sigma < 1),
                reflecting aversion to estimation risk.

        Returns:
            The credibility factor :math:`C_\Sigma` for scaling the portfolio variance.

        Raises:
            RuntimeError: If posterior parameters have not been computed.
            ValueError: If `p_sigma` is not in (0,1) or internal terms are invalid.
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
            # This should not happen with valid inputs but good to check
            warnings.warn(
                f"Argument for sqrt in C_Σ calculation is negative ({term2_arg:f}); result may be NaN.",
                RuntimeWarning
            )
        term2 = np.sqrt(term2_arg)
        C_sigma = term1 + term2
        return C_sigma
