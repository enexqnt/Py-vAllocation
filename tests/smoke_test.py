import unittest
import numpy as np
import pandas as pd
from pyvallocation import probabilities, moments, views, portfolioapi, bayesian
from pyvallocation.utils import projection

class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.T = 100
        self.N = 5
        self.returns_array = np.random.rand(self.T, self.N) - 0.5
        self.returns_df = pd.DataFrame(self.returns_array, columns=[f'Asset_{i}' for i in range(self.N)])
        self.spy_vol_series = pd.Series(np.random.rand(self.T), name='SPY_VOL')
        self.half_life = 50

    def test_generate_uniform_probabilities(self):
        p_uniform = probabilities.generate_uniform_probabilities(self.T)
        self.assertEqual(len(p_uniform), self.T)
        self.assertAlmostEqual(np.sum(p_uniform), 1.0)
        self.assertTrue(np.all(p_uniform > 0))
        self.assertTrue(np.allclose(p_uniform, np.full(self.T, 1/self.T)))

    def test_generate_exp_decay_probabilities(self):
        p_exp = probabilities.generate_exp_decay_probabilities(self.T, self.half_life)
        self.assertEqual(len(p_exp), self.T)
        self.assertAlmostEqual(np.sum(p_exp), 1.0)
        self.assertTrue(np.all(p_exp > 0))



    def test_generate_gaussian_kernel_probabilities(self):
        p_gk = probabilities.generate_gaussian_kernel_probabilities(self.spy_vol_series)
        self.assertEqual(len(p_gk), self.T)
        self.assertAlmostEqual(np.sum(p_gk), 1.0)
        self.assertTrue(np.all(p_gk >= 0))

    def test_compute_effective_number_scenarios(self):
        p_uniform = probabilities.generate_uniform_probabilities(self.T)
        ens_uniform = probabilities.compute_effective_number_scenarios(p_uniform)
        self.assertAlmostEqual(ens_uniform, self.T)

        p_exp = probabilities.generate_exp_decay_probabilities(self.T, self.half_life)
        ens_exp = probabilities.compute_effective_number_scenarios(p_exp)
        self.assertTrue(1 <= ens_exp <= self.T)

        p_concentrated = np.zeros(self.T)
        p_concentrated[0] = 1.0
        ens_concentrated = probabilities.compute_effective_number_scenarios(p_concentrated)
        self.assertAlmostEqual(ens_concentrated, 1.0)

    def test_estimate_sample_moments(self):
        p_uniform = probabilities.generate_uniform_probabilities(self.T)
        mu, cov = moments.estimate_sample_moments(self.returns_df, p_uniform)
        self.assertEqual(mu.shape, (self.N,))
        self.assertEqual(cov.shape, (self.N, self.N))
        self.assertIsInstance(mu, pd.Series)
        self.assertIsInstance(cov, pd.DataFrame)

        mu_np, cov_np = moments.estimate_sample_moments(self.returns_array, p_uniform)
        self.assertEqual(mu_np.shape, (self.N,))
        self.assertEqual(cov_np.shape, (self.N, self.N))
        self.assertIsInstance(mu_np, np.ndarray)
        self.assertIsInstance(cov_np, np.ndarray)


    def test_shrink_mean_jorion(self):
        p_uniform = probabilities.generate_uniform_probabilities(self.T)
        mu, cov = moments.estimate_sample_moments(self.returns_df, p_uniform)
        
        if self.T <= self.N + 2:
            original_T = self.T
            self.T = self.N + 3
            temp_returns_df = pd.DataFrame(np.random.rand(self.T, self.N) - 0.5)
            p_uniform_temp = probabilities.generate_uniform_probabilities(self.T)
            mu_temp, cov_temp = moments.estimate_sample_moments(temp_returns_df, p_uniform_temp)
            mu_shrunk = moments.shrink_mean_jorion(mu_temp, cov_temp, self.T)
            self.T = original_T
        else:
            mu_shrunk = moments.shrink_mean_jorion(mu, cov, self.T)

        self.assertEqual(mu_shrunk.shape, (self.N,))
        self.assertIsInstance(mu_shrunk, type(mu))


    def test_shrink_covariance_ledoit_wolf_constant_correlation(self):
        p_uniform = probabilities.generate_uniform_probabilities(self.T)
        _ , cov = moments.estimate_sample_moments(self.returns_df, p_uniform)
        
        if self.T < self.N:
            original_T = self.T
            self.T = self.N + 1
            temp_returns_df = pd.DataFrame(np.random.rand(self.T, self.N) - 0.5, columns=self.returns_df.columns)
            p_uniform_temp = probabilities.generate_uniform_probabilities(self.T)
            _ , cov_temp = moments.estimate_sample_moments(temp_returns_df, p_uniform_temp)
            cov_shrunk = moments.shrink_covariance_ledoit_wolf(temp_returns_df, cov_temp, target='constant_correlation')
            self.T = original_T
        else:
            cov_shrunk = moments.shrink_covariance_ledoit_wolf(self.returns_df, cov, target='constant_correlation')

        self.assertEqual(cov_shrunk.shape, (self.N, self.N))
        self.assertIsInstance(cov_shrunk, pd.DataFrame)

    def test_shrink_covariance_ledoit_wolf_identity(self):
        p_uniform = probabilities.generate_uniform_probabilities(self.T)
        _ , cov = moments.estimate_sample_moments(self.returns_df, p_uniform)

        if self.T < self.N:
            original_T = self.T
            self.T = self.N + 1
            temp_returns_df = pd.DataFrame(np.random.rand(self.T, self.N) - 0.5, columns=self.returns_df.columns)
            p_uniform_temp = probabilities.generate_uniform_probabilities(self.T)
            _ , cov_temp = moments.estimate_sample_moments(temp_returns_df, p_uniform_temp)
            cov_shrunk = moments.shrink_covariance_ledoit_wolf(temp_returns_df, cov_temp, target='identity')
            self.T = original_T
        else:
            cov_shrunk = moments.shrink_covariance_ledoit_wolf(self.returns_df, cov, target='identity')
            
        self.assertEqual(cov_shrunk.shape, (self.N, self.N))
        self.assertIsInstance(cov_shrunk, pd.DataFrame)

    def test_black_litterman_processor(self):
        prior_mean = pd.Series(np.random.rand(self.N) / 100, index=self.returns_df.columns)
        prior_cov = pd.DataFrame(np.cov(self.returns_array.T) + np.eye(self.N)*0.0001, index=self.returns_df.columns, columns=self.returns_df.columns)
        
        eigenvalues = np.linalg.eigvalsh(prior_cov)
        if np.any(eigenvalues < 0):
            prior_cov = prior_cov + np.eye(self.N) * (abs(np.min(eigenvalues)) + 1e-6)


        mean_views = {self.returns_df.columns[0]: 0.001}
        view_confidences = 0.8

        blp = views.BlackLittermanProcessor(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            mean_views=mean_views,
            view_confidences=view_confidences,
            omega='idzorek',
            idzorek_use_tau=False
        )
        posterior_mean, posterior_cov = blp.get_posterior()

        self.assertEqual(posterior_mean.shape, (self.N,))
        self.assertEqual(posterior_cov.shape, (self.N, self.N))
        self.assertIsInstance(posterior_mean, pd.Series)
        self.assertIsInstance(posterior_cov, pd.DataFrame)

    def test_project_mean_covariance(self):
        mu = pd.Series(np.random.rand(self.N) / 252, index=self.returns_df.columns)
        cov = pd.DataFrame(np.cov(self.returns_array.T)/252 + np.eye(self.N)*0.00001, index=self.returns_df.columns, columns=self.returns_df.columns)
        investment_horizon = 20

        mu_hor, cov_hor = projection.project_mean_covariance(mu, cov, investment_horizon)
        self.assertEqual(mu_hor.shape, (self.N,))
        self.assertEqual(cov_hor.shape, (self.N, self.N))
        self.assertTrue(np.all(mu_hor >= mu * investment_horizon * 0.9) and np.all(mu_hor <= mu * investment_horizon * 1.1) if not mu.empty else True)

    def test_log2simple(self):
        log_mu = pd.Series(np.random.rand(self.N) * 0.01, index=self.returns_df.columns)
        log_cov = pd.DataFrame(np.random.rand(self.N, self.N) * 0.001, index=self.returns_df.columns, columns=self.returns_df.columns)
        log_cov = (log_cov + log_cov.T)/2 + np.eye(self.N)*1e-5

        simple_mu, simple_cov = projection.log2simple(log_mu, log_cov)
        self.assertEqual(simple_mu.shape, (self.N,))
        self.assertEqual(simple_cov.shape, (self.N, self.N))
        self.assertIsInstance(simple_mu, pd.Series)
        self.assertIsInstance(simple_cov, pd.DataFrame)


    def test_assets_distribution(self):
        mu = pd.Series(np.random.rand(self.N) / 100, index=self.returns_df.columns)
        cov = pd.DataFrame(np.cov(self.returns_array.T) + np.eye(self.N)*0.0001, index=self.returns_df.columns, columns=self.returns_df.columns)
        dist = portfolioapi.AssetsDistribution(mu=mu, cov=cov)
        self.assertTrue(np.array_equal(dist.mu, mu))
        if isinstance(dist.cov, pd.DataFrame) and isinstance(cov, pd.DataFrame):
            self.assertTrue(dist.cov.equals(cov))
        else:
            self.assertTrue(np.array_equal(dist.cov, cov))


    def test_portfolio_wrapper_mvo(self):
        mu = pd.Series(np.random.rand(self.N) / 100, index=self.returns_df.columns)
        cov_matrix = np.cov(self.returns_array.T)
        min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eig <= 1e-8:
            cov_matrix += np.eye(self.N) * (abs(min_eig) + 1e-7)

        cov = pd.DataFrame(cov_matrix, index=self.returns_df.columns, columns=self.returns_df.columns)
        
        asset_dist = portfolioapi.AssetsDistribution(mu=mu, cov=cov)
        wrapper = portfolioapi.PortfolioWrapper(asset_dist)

        constraints_to_set = {'long_only': True, 'total_weight': 1.0}
        wrapper.set_constraints(constraints_to_set)

        num_portfolios = 10
        frontier = wrapper.mean_variance_frontier(num_portfolios=num_portfolios)
        self.assertEqual(frontier.weights.shape, (self.N, num_portfolios))
        self.assertEqual(len(frontier.returns), num_portfolios)
        self.assertEqual(len(frontier.risks), num_portfolios)

        w_min_risk, r_min_risk, risk_val_min_risk = frontier.get_min_risk_portfolio()
        self.assertEqual(w_min_risk.shape, (self.N,))
        self.assertIsInstance(r_min_risk, float)
        self.assertIsInstance(risk_val_min_risk, float)

        min_frontier_return = np.min(frontier.returns)
        max_frontier_return = np.max(frontier.returns)
        target_return = (min_frontier_return + max_frontier_return) / 2.0
        
        if max_frontier_return > min_frontier_return:
            w_target, r_target, risk_target = frontier.portfolio_at_return_target(target_return)
            self.assertEqual(w_target.shape, (self.N,))
            self.assertIsInstance(r_target, float)
            self.assertIsInstance(risk_target, float)
            self.assertAlmostEqual(r_target, target_return, places=3)
        else:
            print("Skipping portfolio_at_return_target sub-test as frontier returns are too close.")

    def test_bayesian_niw_posterior(self):
        prior_mean_bl = pd.Series(np.random.rand(self.N) / 100, index=self.returns_df.columns)
        prior_cov_bl_matrix = np.cov(self.returns_array.T) + np.eye(self.N)*0.001
        min_eig_bl = np.min(np.linalg.eigvalsh(prior_cov_bl_matrix))
        if min_eig_bl <= 1e-8:
            prior_cov_bl_matrix += np.eye(self.N) * (abs(min_eig_bl) + 1e-7)
        prior_cov_bl = pd.DataFrame(prior_cov_bl_matrix, index=self.returns_df.columns, columns=self.returns_df.columns)

        kappa_prior = self.T / 2.0
        nu_prior = self.T / 2.0

        niw_updater = bayesian.NIWPosterior(prior_mean_bl, prior_cov_bl, kappa_prior, nu_prior)

        sample_mean = pd.Series(np.random.rand(self.N) / 100, index=self.returns_df.columns)
        sample_cov_matrix = np.cov(self.returns_array.T * 1.1) + np.eye(self.N)*0.001
        min_eig_sample = np.min(np.linalg.eigvalsh(sample_cov_matrix))
        if min_eig_sample <= 1e-8:
            sample_cov_matrix += np.eye(self.N) * (abs(min_eig_sample) + 1e-7)
        sample_cov = pd.DataFrame(sample_cov_matrix, index=self.returns_df.columns, columns=self.returns_df.columns)
        n_obs_update = self.T

        posterior_params = niw_updater.update(sample_mean, sample_cov, n_obs_update)

        self.assertIsNotNone(posterior_params.mu1)
        self.assertIsNotNone(posterior_params.sigma1)
        self.assertEqual(posterior_params.mu1.shape, (self.N,))
        self.assertEqual(posterior_params.sigma1.shape, (self.N, self.N))

        gamma_mu = niw_updater.cred_radius_mu(p_mu=0.9)
        self.assertIsInstance(gamma_mu, float)
        self.assertGreaterEqual(gamma_mu, 0)


    def test_portfolio_wrapper_robust_lambda_frontier(self):
        est_expected_return = pd.Series(np.random.rand(self.N) / 90, index=self.returns_df.columns)
        est_uncertainty_cov_matrix = np.cov(self.returns_array.T * 0.9) + np.eye(self.N)*0.0015
        min_eig_robust = np.min(np.linalg.eigvalsh(est_uncertainty_cov_matrix))
        if min_eig_robust <= 1e-8:
            est_uncertainty_cov_matrix += np.eye(self.N) * (abs(min_eig_robust) + 1e-7)
        est_uncertainty_cov = pd.DataFrame(est_uncertainty_cov_matrix, index=self.returns_df.columns, columns=self.returns_df.columns)

        robust_dist = portfolioapi.AssetsDistribution(mu=est_expected_return, cov=est_uncertainty_cov)
        robust_wrapper = portfolioapi.PortfolioWrapper(robust_dist)
        robust_wrapper.set_constraints({'long_only': True, 'total_weight': 1.0})

        num_portfolios_robust = 10
        lambda_frontier = robust_wrapper.robust_lambda_frontier(num_portfolios=num_portfolios_robust, max_lambda=1.0)

        self.assertEqual(lambda_frontier.weights.shape, (self.N, num_portfolios_robust))
        self.assertEqual(len(lambda_frontier.returns), num_portfolios_robust)
        self.assertEqual(len(lambda_frontier.risks), num_portfolios_robust)

        w_min_est_risk, r_min_est_risk, est_risk_min = lambda_frontier.get_min_risk_portfolio()
        self.assertEqual(w_min_est_risk.shape, (self.N,))
        self.assertIsInstance(r_min_est_risk, float)
        self.assertIsInstance(est_risk_min, float)


    def test_portfolio_wrapper_solve_robust_gamma_portfolio(self):
        est_expected_return = pd.Series(np.random.rand(self.N) / 95, index=self.returns_df.columns)
        est_uncertainty_cov_matrix = np.cov(self.returns_array.T * 0.95) + np.eye(self.N)*0.0012
        min_eig_gamma = np.min(np.linalg.eigvalsh(est_uncertainty_cov_matrix))
        if min_eig_gamma <= 1e-8:
            est_uncertainty_cov_matrix += np.eye(self.N) * (abs(min_eig_gamma) + 1e-7)
        est_uncertainty_cov = pd.DataFrame(est_uncertainty_cov_matrix, index=self.returns_df.columns, columns=self.returns_df.columns)

        robust_dist = portfolioapi.AssetsDistribution(mu=est_expected_return, cov=est_uncertainty_cov)
        robust_wrapper = portfolioapi.PortfolioWrapper(robust_dist)
        robust_wrapper.set_constraints({'long_only': True, 'total_weight': 1.0})

        gamma_mu_val = 0.05
        gamma_sigma_sq_val = (0.4)**2

        w_gamma, r_gamma, est_risk_gamma = robust_wrapper.solve_robust_gamma_portfolio(
            gamma_mu=gamma_mu_val,
            gamma_sigma_sq=gamma_sigma_sq_val
        )

        self.assertEqual(w_gamma.shape, (self.N,))
        self.assertIsInstance(r_gamma, float)
        self.assertIsInstance(est_risk_gamma, float)
        self.assertLessEqual(est_risk_gamma, gamma_sigma_sq_val + 1e-5)

    def test_project_scenarios(self):
        p_exp = probabilities.generate_exp_decay_probabilities(self.T, self.half_life)
        investment_horizon = 10
        
        projected_scenarios_df = projection.project_scenarios(
            self.returns_df, 
            investment_horizon, 
            p_exp, 
            n_simulations=self.T 
        )
        self.assertIsInstance(projected_scenarios_df, pd.DataFrame)
        self.assertEqual(projected_scenarios_df.shape, (self.T, self.N)) 
        self.assertEqual(list(projected_scenarios_df.columns), list(self.returns_df.columns))

        projected_scenarios_np = projection.project_scenarios(
            self.returns_array, 
            investment_horizon, 
            p_exp, 
            n_simulations=self.T
        )
        self.assertIsInstance(projected_scenarios_np, np.ndarray)
        self.assertEqual(projected_scenarios_np.shape, (self.T, self.N))

    def test_convert_scenarios_compound_to_simple(self):
        compound_scenarios = np.random.randn(self.T, self.N) * 0.01 
        
        simple_scenarios_np = projection.convert_scenarios_compound_to_simple(compound_scenarios)
        self.assertIsInstance(simple_scenarios_np, np.ndarray)
        self.assertEqual(simple_scenarios_np.shape, compound_scenarios.shape)
        self.assertTrue(np.all(simple_scenarios_np > -1))

        compound_scenarios_df = pd.DataFrame(compound_scenarios, columns=self.returns_df.columns)
        simple_scenarios_df = projection.convert_scenarios_compound_to_simple(compound_scenarios_df)
        self.assertIsInstance(simple_scenarios_df, pd.DataFrame)
        self.assertEqual(simple_scenarios_df.shape, compound_scenarios_df.shape)
        self.assertEqual(list(simple_scenarios_df.columns), list(compound_scenarios_df.columns))
        self.assertTrue(np.all(simple_scenarios_df.values > -1))


    def test_flexible_views_processor_mean_views(self):
        p_prior = probabilities.generate_uniform_probabilities(self.T)
        mean_views = {self.returns_df.columns[0]: ('>', 0.0005), self.returns_df.columns[1]: 0.0001}
        
        fvp = views.FlexibleViewsProcessor(
            prior_returns=self.returns_df,
            prior_probabilities=p_prior,
            mean_views=mean_views
        )
        q_posterior = fvp.get_posterior_probabilities()

        self.assertEqual(q_posterior.shape, (self.T, 1))
        self.assertAlmostEqual(np.sum(q_posterior), 1.0, places=5)
        self.assertTrue(np.all(q_posterior >= 0))

    def test_flexible_views_processor_corr_vol_views(self):
        p_prior = probabilities.generate_exp_decay_probabilities(self.T, self.half_life)
        if self.N < 2:
            self.skipTest("Need at least 2 assets for correlation views test.")
            return

        corr_views = {(self.returns_df.columns[0], self.returns_df.columns[1]): ('<', 0.1)}
        vol_views = {self.returns_df.columns[0]: ('>', 0.01)}

        fvp = views.FlexibleViewsProcessor(
            prior_returns=self.returns_df,
            prior_probabilities=p_prior,
            corr_views=corr_views,
            vol_views=vol_views
        )
        q_posterior = fvp.get_posterior_probabilities()

        self.assertEqual(q_posterior.shape, (self.T, 1))
        self.assertAlmostEqual(np.sum(q_posterior), 1.0)
        self.assertTrue(np.all(q_posterior >= 0))

    def test_flexible_views_processor_mean_cov_prior(self):
        prior_mean = pd.Series(np.random.rand(self.N) / 100, index=self.returns_df.columns)
        prior_cov_matrix = np.cov(self.returns_array.T) + np.eye(self.N)*0.0001
        min_eig = np.min(np.linalg.eigvalsh(prior_cov_matrix))
        if min_eig <= 1e-8:
            prior_cov_matrix += np.eye(self.N) * (abs(min_eig) + 1e-7)
        prior_cov = pd.DataFrame(prior_cov_matrix, index=self.returns_df.columns, columns=self.returns_df.columns)

        mean_views = {self.returns_df.columns[0]: 0.0008}
        
        fvp = views.FlexibleViewsProcessor(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            mean_views=mean_views,
        )
        mu_posterior, cov_posterior = fvp.get_posterior()

        self.assertEqual(mu_posterior.shape, (self.N,))
        self.assertEqual(cov_posterior.shape, (self.N, self.N))
        self.assertIsInstance(mu_posterior, pd.Series)
        self.assertIsInstance(cov_posterior, pd.DataFrame)
        
        eigenvalues_posterior = np.linalg.eigvalsh(cov_posterior)
        self.assertTrue(np.all(eigenvalues_posterior >= -1e-8))

    def test_portfolio_wrapper_mean_cvar_frontier(self):
        scenarios = np.random.randn(self.T, self.N) * 0.02
        scenarios_df = pd.DataFrame(scenarios, columns=self.returns_df.columns)

        cvar_dist = portfolioapi.AssetsDistribution(scenarios=scenarios_df)
        cvar_wrapper = portfolioapi.PortfolioWrapper(cvar_dist)
        cvar_wrapper.set_constraints({'long_only': True, 'total_weight': 1.0})

        num_portfolios_cvar = 10
        alpha_cvar = 0.05
        cvar_frontier = cvar_wrapper.mean_cvar_frontier(num_portfolios=num_portfolios_cvar, alpha=alpha_cvar)

        self.assertEqual(cvar_frontier.weights.shape, (self.N, num_portfolios_cvar))
        self.assertEqual(len(cvar_frontier.returns), num_portfolios_cvar)
        self.assertEqual(len(cvar_frontier.risks), num_portfolios_cvar)
        self.assertEqual(cvar_frontier.risk_measure, f"CVaR (alpha={alpha_cvar:.2f})")

        w_min_cvar, r_min_cvar, cvar_val_min = cvar_frontier.get_min_risk_portfolio()
        self.assertEqual(w_min_cvar.shape, (self.N,))
        self.assertIsInstance(r_min_cvar, float)
        self.assertIsInstance(cvar_val_min, float)

        min_cvar_on_frontier = np.min(cvar_frontier.risks)
        max_cvar_on_frontier = np.max(cvar_frontier.risks)
        target_cvar = (min_cvar_on_frontier + max_cvar_on_frontier) / 2.0
        if max_cvar_on_frontier > min_cvar_on_frontier + 1e-6:
            w_target_cvar, r_target_cvar, cvar_at_target = cvar_frontier.portfolio_at_risk_target(max_risk=target_cvar)
            self.assertEqual(w_target_cvar.shape, (self.N,))
            self.assertIsInstance(r_target_cvar, float)
            self.assertIsInstance(cvar_at_target, float)
            self.assertLessEqual(cvar_at_target, target_cvar + 1e-5)

        min_ret_on_frontier = np.min(cvar_frontier.returns)
        max_ret_on_frontier = np.max(cvar_frontier.returns)
        target_return_cvar = (min_ret_on_frontier + max_ret_on_frontier) / 2.0
        if max_ret_on_frontier > min_ret_on_frontier + 1e-6:
            w_target_ret, r_at_target_ret, cvar_at_target_ret = cvar_frontier.portfolio_at_return_target(min_return=target_return_cvar)
            self.assertEqual(w_target_ret.shape, (self.N,))
            self.assertIsInstance(r_at_target_ret, float)
            self.assertIsInstance(cvar_at_target_ret, float)
            self.assertGreaterEqual(r_at_target_ret, target_return_cvar - 1e-5)


if __name__ == '__main__':
    unittest.main()

