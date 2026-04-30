"""Adaptive CoCoLasso aligned with the covariance-form CoCoLasso pipeline."""

from __future__ import annotations

import numpy as np

from .CoCoLasso import (
    _as_2d_x_y,
    _preprocess,
    _ratio_matrix_missing,
    corrected_covariance,
    corrected_rho,
    lasso_covariance,
    project_covariance,
    Noise,
    Penalty,
    ProjectionMode,
)


class ACoCoLasso:
    """Adaptive CoCoLasso using weighted covariance-form Lasso.

    This follows the same preprocessing and covariance correction steps as
    CoCoLasso, then applies adaptive weights derived from an external
    initial coefficient estimate.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.0,
        init_coef: np.ndarray | None = None,
        *,
        center_z: bool = True,
        scale_z: bool = True,
        center_y: bool = True,
        scale_y: bool = True,
        mu: float = 10.0,
        tau: float | None = None,
        etol: float = 1e-4,
        opt_tol: float = 1e-5,
        epsilon: float = 1e-8,
        noise: Noise = "additive",
        penalty: Penalty = "lasso",
        mode: ProjectionMode = "ADMM",
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.init_coef = init_coef
        self.center_z = center_z
        self.scale_z = scale_z
        self.center_y = center_y
        self.scale_y = scale_y
        self.mu = mu
        self.tau = tau
        self.etol = etol
        self.opt_tol = opt_tol
        self.epsilon = epsilon
        self.noise = noise
        self.penalty = penalty
        self.mode = mode

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.weights_: np.ndarray | None = None
        self.init_coef_: np.ndarray | None = None
        self.mean_z_: np.ndarray | None = None
        self.sd_z_: np.ndarray | None = None
        self.mean_y_: float | None = None
        self.sd_y_: float | None = None

    def _scaled_init_coef(self, init_coef: np.ndarray, sd_z: np.ndarray, sd_y: float) -> np.ndarray:
        scale = np.ones_like(init_coef, dtype=float)
        if self.scale_z:
            scale = scale * sd_z
        if self.scale_y:
            scale = scale / sd_y
        return init_coef * scale

    def _rescale_coef(self, beta_scaled: np.ndarray, sd_z: np.ndarray, sd_y: float) -> np.ndarray:
        coef = beta_scaled.copy()
        if self.scale_z:
            coef = coef / sd_z
        if self.scale_y:
            coef = coef * sd_y
        return coef

    def fit(self, z: np.ndarray, y: np.ndarray) -> "ACoCoLasso":
        z, y = _as_2d_x_y(z, y)
        if self.noise == "missing" and not self.center_z:
            raise ValueError("center_z=True is required for missing data.")

        ratio_matrix = _ratio_matrix_missing(z) if self.noise == "missing" else None
        z_work, y_work, mean_z, sd_z, mean_y, sd_y = _preprocess(
            z, y, self.center_z, self.scale_z, self.center_y, self.scale_y
        )

        if self.init_coef is None:
            raise ValueError("ACoCoLasso requires externally provided init_coef.")
        init_coef = np.asarray(self.init_coef, dtype=float).reshape(-1)
        if init_coef.size != z.shape[1]:
            raise ValueError(f"init_coef length must be {z.shape[1]}, got {init_coef.size}")
        if not np.all(np.isfinite(init_coef)):
            raise ValueError("init_coef must contain only finite values")
        self.init_coef_ = init_coef.copy()

        init_scaled = self._scaled_init_coef(init_coef, sd_z, sd_y)
        weights = 1.0 / (np.abs(init_scaled) + self.epsilon) ** self.gamma
        self.weights_ = weights.copy()
        inv_weights = 1.0 / weights

        sigma_hat = corrected_covariance(
            z_work, self.noise, tau=self.tau, ratio_matrix=ratio_matrix
        )
        rho_hat = corrected_rho(z_work, y_work, self.noise, ratio_matrix=ratio_matrix)
        sigma_tilde = project_covariance(
            sigma_hat, mode=self.mode, ratio=ratio_matrix, mu=self.mu, etol=self.etol
        )

        sigma_weighted = (inv_weights[:, None] * sigma_tilde) * inv_weights[None, :]
        rho_weighted = rho_hat * inv_weights

        beta_weighted = lasso_covariance(
            sigma_weighted,
            rho_weighted,
            self.alpha,
            penalty=self.penalty,
            opt_tol=self.opt_tol,
        )
        beta_scaled = beta_weighted * inv_weights

        coef = self._rescale_coef(beta_scaled, sd_z, sd_y)
        intercept = 0.0
        if self.center_y:
            intercept += mean_y
        if self.center_z:
            intercept -= float(mean_z @ coef)

        self.coef_ = coef
        self.intercept_ = intercept
        self.mean_z_ = mean_z
        self.sd_z_ = sd_z
        self.mean_y_ = mean_y
        self.sd_y_ = sd_y
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return np.asarray(z, dtype=float) @ self.coef_ + float(self.intercept_)


__all__ = ["ACoCoLasso"]
