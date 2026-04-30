"""CoCoLasso implementation based on the python/cocolasso.py port.

This module keeps the same mathematical decomposition as the reference R
implementation: corrected covariance, PSD projection, then covariance-form
Lasso solved by coordinate descent. A thin class wrapper is provided for
compatibility with the src.models API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Noise = Literal["additive", "missing"]
Penalty = Literal["lasso", "SCAD"]
ProjectionMode = Literal["ADMM", "HM"]


@dataclass
class CocoResult:
    lambda_opt: float
    lambda_sd: float
    beta_opt: np.ndarray
    beta_sd: np.ndarray
    data_error: np.ndarray
    data_beta: np.ndarray
    early_stopping: int
    feature_names: list[str] | None
    mean_z: np.ndarray
    sd_z: np.ndarray
    mean_y: float
    sd_y: float
    center_z: bool
    scale_z: bool
    center_y: bool
    scale_y: bool

    def coef(self, s: float | None = None) -> np.ndarray:
        """Return coefficients for a requested lambda or the whole path."""
        if s is None:
            return self.data_beta[:, 1:].T
        idx = int(np.argmin(np.abs(self.data_beta[:, 0] - s)))
        return self.data_beta[idx, 1:].copy()

    def predict(self, newx: np.ndarray, lambda_pred: float | None = None) -> np.ndarray:
        """Predict on the original data scale."""
        x = np.asarray(newx, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        beta = self.beta_sd
        if lambda_pred is not None:
            idx = int(np.argmin(np.abs(self.data_beta[:, 0] - lambda_pred)))
            beta = self.data_beta[idx, 1:]

        x_work = x.copy()
        if self.center_z:
            x_work = x_work - self.mean_z
        if self.scale_z:
            x_work = x_work / self.sd_z

        y_work = x_work @ beta
        if self.scale_y:
            y_work = y_work * self.sd_y
        if self.center_y:
            y_work = y_work + self.mean_y
        return y_work


def _as_2d_x_y(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if z.ndim != 2:
        raise ValueError("Z must be a 2D array.")
    if y.ndim != 1 or y.shape[0] != z.shape[0]:
        raise ValueError("y must have one value per row of Z.")
    if np.isnan(y).any():
        raise ValueError("y contains NaN values; remove them before fitting.")
    return z, y


def _nanstd_ddof1(x: np.ndarray, axis: int = 0) -> np.ndarray:
    sd = np.nanstd(x, axis=axis, ddof=1)
    return np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)


def _preprocess(
    z: np.ndarray,
    y: np.ndarray,
    center_z: bool,
    scale_z: bool,
    center_y: bool,
    scale_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    mean_z = np.nanmean(z, axis=0)
    sd_z = _nanstd_ddof1(z, axis=0)
    z_work = z.copy()

    if center_z:
        z_work = z_work - mean_z
    z_work = np.where(np.isnan(z_work), 0.0, z_work)
    if scale_z:
        z_work = z_work / sd_z

    mean_y = float(np.mean(y))
    sd_y = float(np.std(y, ddof=1))
    if not np.isfinite(sd_y) or sd_y == 0:
        sd_y = 1.0
    y_work = y.copy()
    if center_y:
        y_work = y_work - mean_y
    if scale_y:
        y_work = y_work / sd_y
    return z_work, y_work, mean_z, sd_z, mean_y, sd_y


def _ratio_matrix_missing(z: np.ndarray) -> np.ndarray:
    observed = (~np.isnan(z)).astype(float)
    ratio = observed.T @ observed
    ratio = ratio.astype(float) / z.shape[0]
    if np.any(np.diag(ratio) == 0):
        raise ValueError("At least one column is completely missing.")
    if np.any(ratio == 0):
        raise ValueError("Some column pairs are never jointly observed.")
    return ratio


def _logspace_desc(max_value: float, min_value: float, steps: int) -> np.ndarray:
    if max_value <= 0:
        return np.zeros(steps)
    return np.exp(np.linspace(np.log(max_value), np.log(min_value), steps))


def l1_projection(v: np.ndarray, radius: float) -> np.ndarray:
    """Project a vector onto the L1 ball of the requested radius."""
    if radius <= 0:
        raise ValueError("radius must be positive.")
    v = np.asarray(v, dtype=float)
    if np.linalg.norm(v, ord=1) <= radius:
        return v.copy()

    u = np.sort(np.abs(v))[::-1]
    sv = np.cumsum(u)
    idx = np.nonzero(u > (sv - radius) / np.arange(1, len(u) + 1))[0]
    rho = idx[-1]
    theta = max(0.0, (sv[rho] - radius) / (rho + 1))
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


def admm_proj(
    mat: np.ndarray,
    epsilon: float = 1e-4,
    mu: float = 10.0,
    max_iter: int = 1000,
    etol: float = 1e-4,
    etol_distance: float = 1e-4,
) -> np.ndarray:
    """Nearest PSD projection used by CoCoLasso, following R/ADMM_proj.R."""
    mat = np.asarray(mat, dtype=float)
    p = mat.shape[0]
    r = np.diag(np.diag(mat)).copy()
    s = np.zeros((p, p), dtype=float)
    lagrange = np.zeros((p, p), dtype=float)
    tril = np.tril_indices(p)

    for itr in range(max_iter):
        r_prev = r.copy()
        s_prev = s.copy()

        w = mat + s + mu * lagrange
        eigvals, eigvecs = np.linalg.eigh((w + w.T) / 2.0)
        r = eigvecs @ np.diag(np.maximum(eigvals, epsilon)) @ eigvecs.T
        r = (r + r.T) / 2.0

        m = r - mat - mu * lagrange
        s = np.zeros_like(mat)
        s[tril] = m[tril] - l1_projection(m[tril], radius=mu / 2.0)
        s = s + np.tril(s, -1).T

        lagrange = lagrange - (r - s - mat) / mu

        primal = np.max(np.abs(r - s - mat))
        stopped_by_residuals = (
            np.max(np.abs(r - r_prev)) < etol
            and np.max(np.abs(s - s_prev)) < etol
            and primal < etol
        )
        stopped_by_distance = abs(
            np.max(np.abs(r_prev - mat)) - np.max(np.abs(r - mat))
        ) < etol_distance
        if stopped_by_residuals or stopped_by_distance:
            break
        if itr > 0 and itr % 20 == 0:
            mu = mu / 2.0
    return r


def hm_proj(
    sigma_hat: np.ndarray,
    ratio: np.ndarray | None = None,
    a: float = 1.0,
    max_iter: int = 1000,
    epsilon: float = 1e-4,
    mu: float = 10.0,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Frobenius-norm HM-Lasso projection, following R/HM_proj.R."""
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    n = sigma_hat.shape[0]
    weights = np.ones((n, n)) if ratio is None else np.asarray(ratio, dtype=float) ** a
    ak = sigma_hat.copy()
    bk = np.zeros_like(sigma_hat)
    lk = np.zeros_like(sigma_hat)

    for _ in range(max_iter):
        a_mat = bk + sigma_hat + mu * lk
        eigvals, eigvecs = np.linalg.eigh((a_mat + a_mat.T) / 2.0)
        ak_next = eigvecs @ np.diag(np.maximum(eigvals, epsilon)) @ eigvecs.T
        ak_next = (ak_next + ak_next.T) / 2.0
        bk_next = (ak_next - sigma_hat - mu * lk) / (mu * weights * weights + 1.0)
        lk_next = lk - (ak_next - bk_next - sigma_hat) / mu
        if max(
            np.max(np.abs(ak_next - ak)),
            np.max(np.abs(bk_next - bk)),
            np.max(np.abs(lk_next - lk)),
        ) < tolerance:
            ak = ak_next
            break
        ak, bk, lk = ak_next, bk_next, lk_next
    return ak


def project_covariance(
    cov: np.ndarray,
    mode: ProjectionMode = "ADMM",
    ratio: np.ndarray | None = None,
    mu: float = 10.0,
    etol: float = 1e-4,
) -> np.ndarray:
    if mode == "ADMM":
        return admm_proj(cov, mu=mu, etol=etol)
    if mode == "HM":
        return hm_proj(cov, ratio=ratio, mu=mu, tolerance=etol)
    raise ValueError("mode must be 'ADMM' or 'HM'.")


def corrected_covariance(
    z: np.ndarray,
    noise: Noise,
    tau: float | None = None,
    ratio_matrix: np.ndarray | None = None,
) -> np.ndarray:
    n, p = z.shape
    cov = (z.T @ z) / n
    if noise == "additive":
        if tau is None:
            raise ValueError("tau is required for additive noise.")
        return cov - tau**2 * np.eye(p)
    if noise == "missing":
        if ratio_matrix is None:
            raise ValueError("ratio_matrix is required for missing noise.")
        return cov / ratio_matrix
    raise ValueError("noise must be 'additive' or 'missing'.")


def corrected_rho(
    z: np.ndarray,
    y: np.ndarray,
    noise: Noise,
    ratio_matrix: np.ndarray | None = None,
) -> np.ndarray:
    rho = z.T @ y / z.shape[0]
    if noise == "missing":
        if ratio_matrix is None:
            raise ValueError("ratio_matrix is required for missing noise.")
        rho = rho / np.diag(ratio_matrix)
    return rho


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def lasso_covariance(
    sigma: np.ndarray,
    rho: np.ndarray,
    lam: float,
    beta_start: np.ndarray | None = None,
    penalty: Penalty = "lasso",
    max_iter: int = 1000,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
) -> np.ndarray:
    """Solve 0.5 beta' Sigma beta - rho' beta + lambda ||beta||_1."""
    sigma = np.asarray(sigma, dtype=float)
    rho = np.asarray(rho, dtype=float).reshape(-1)
    p = sigma.shape[0]
    beta = np.zeros(p, dtype=float) if beta_start is None else beta_start.astype(float).copy()
    residual_product = sigma @ beta
    lambda0 = float(lam)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            diag = sigma[j, j]
            if not np.isfinite(diag) or diag <= 0:
                beta[j] = 0.0
                continue

            lambda_j = lambda0
            if penalty == "SCAD":
                a = 3.7
                abs_beta = abs(beta[j])
                if abs_beta <= lambda0:
                    weight = 1.0
                elif abs_beta <= a * lambda0:
                    weight = (a * lambda0 - abs_beta) / (lambda0 * (a - 1.0))
                else:
                    weight = 0.0
                lambda_j = weight * lambda0

            partial = residual_product[j] - diag * beta[j]
            proposal = _soft_threshold(rho[j] - partial, lambda_j) / diag
            if not np.isfinite(proposal):
                proposal = 0.0
            delta = proposal - beta[j]
            if delta != 0:
                beta[j] = proposal
                residual_product = residual_product + sigma[:, j] * delta

        if np.sum(np.abs(beta - beta_old)) < opt_tol:
            break

    beta[np.abs(beta) < zero_threshold] = 0.0
    return beta


def _make_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k < 2:
        raise ValueError("K must be at least 2.")
    if n % k != 0:
        raise ValueError("K should divide n, matching the original R package.")
    labels = np.repeat(np.arange(k), n // k)
    return rng.permutation(labels)


def _cv_matrices(
    z: np.ndarray,
    y: np.ndarray,
    k: int,
    noise: Noise,
    tau: float | None,
    ratio_matrix: np.ndarray | None,
    mode: ProjectionMode,
    mu: float,
    etol: float,
    rng: np.random.Generator,
):
    n, p = z.shape
    folds = _make_folds(n, k, rng)

    sigma_global = project_covariance(
        corrected_covariance(z, noise, tau=tau, ratio_matrix=ratio_matrix),
        mode=mode,
        ratio=ratio_matrix,
        mu=mu,
        etol=etol,
    )
    rho_global = corrected_rho(z, y, noise, ratio_matrix=ratio_matrix)

    sigma_train = []
    sigma_test = []
    rho_train = []
    rho_test = []
    for fold in range(k):
        test_idx = folds == fold
        train_idx = ~test_idx
        z_train = z[train_idx]
        y_train = y[train_idx]
        z_test = z[test_idx]
        y_test = y[test_idx]

        sigma_train.append(
            project_covariance(
                corrected_covariance(z_train, noise, tau=tau, ratio_matrix=ratio_matrix),
                mode=mode,
                ratio=ratio_matrix,
                mu=mu,
                etol=etol,
            )
        )
        sigma_test.append(
            project_covariance(
                corrected_covariance(z_test, noise, tau=tau, ratio_matrix=ratio_matrix),
                mode=mode,
                ratio=ratio_matrix,
                mu=mu,
                etol=etol,
            )
        )
        rho_train.append(corrected_rho(z_train, y_train, noise, ratio_matrix=ratio_matrix))
        rho_test.append(corrected_rho(z_test, y_test, noise, ratio_matrix=ratio_matrix))

    return sigma_global, rho_global, sigma_train, sigma_test, rho_train, rho_test, folds


def _one_se_lambda(
    data_error: np.ndarray,
    beta_path: np.ndarray,
    best_error: float,
    lambda_opt: float,
) -> tuple[float, np.ndarray]:
    best_idx = int(np.argmin(data_error[:, 1]))
    threshold = best_error + data_error[best_idx, 4]
    eligible = np.where((data_error[:, 1] <= threshold) & (data_error[:, 0] >= lambda_opt))[0]
    idx = int(eligible[0]) if len(eligible) else best_idx
    return float(data_error[idx, 0]), beta_path[idx].copy()


def pathwise_coordinate_descent(
    z: np.ndarray,
    y: np.ndarray,
    *,
    center_z: bool = True,
    scale_z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: float | None = None,
    steps: int = 100,
    k: int = 4,
    mu: float = 10.0,
    tau: float | None = None,
    etol: float = 1e-4,
    opt_tol: float = 1e-5,
    early_stopping_max: int = 10,
    noise: Noise = "additive",
    penalty: Penalty = "lasso",
    mode: ProjectionMode = "ADMM",
    random_state: int | None = None,
    feature_names: list[str] | None = None,
) -> CocoResult:
    z, y = _as_2d_x_y(z, y)
    n, p = z.shape
    if lambda_factor is None:
        lambda_factor = 0.01 if n < p else 0.001
    if not 0 < lambda_factor < 1:
        raise ValueError("lambda_factor must be in (0, 1).")
    if noise == "missing" and not center_z:
        raise ValueError("center_z=True is required for missing data.")

    ratio_matrix = _ratio_matrix_missing(z) if noise == "missing" else None
    z_work, y_work, mean_z, sd_z, mean_y, sd_y = _preprocess(
        z, y, center_z, scale_z, center_y, scale_y
    )

    rho_for_lambda = corrected_rho(z_work, y_work, noise, ratio_matrix=ratio_matrix)
    lambda_max = float(np.max(np.abs(rho_for_lambda)))
    lambda_values = _logspace_desc(lambda_max, lambda_factor * lambda_max, steps)

    rng = np.random.default_rng(random_state)
    sigma_global, rho_global, sigma_train, sigma_test, rho_train, rho_test, _ = _cv_matrices(
        z_work, y_work, k, noise, tau, ratio_matrix, mode, mu, etol, rng
    )

    beta_start = np.zeros(p)
    beta_path = np.zeros((steps, p))
    error_rows = np.zeros((steps, 5))
    best_error = np.inf
    lambda_opt = lambda_values[0]
    beta_opt = beta_start.copy()
    early_stopping = steps
    higher_error_count = 0
    previous_error = np.inf

    for i, lam in enumerate(lambda_values):
        fold_errors = []
        for fold in range(k):
            beta_cv = lasso_covariance(
                sigma_train[fold], rho_train[fold], lam, beta_start, penalty=penalty
            )
            error = beta_cv @ sigma_test[fold] @ beta_cv - 2.0 * rho_test[fold] @ beta_cv
            fold_errors.append(float(error))
        fold_errors = np.asarray(fold_errors)
        error = float(np.mean(fold_errors))

        beta_start = lasso_covariance(
            sigma_global, rho_global, lam, beta_start, penalty=penalty
        )
        beta_path[i] = beta_start
        error_rows[i] = [
            lam,
            error,
            float(np.quantile(fold_errors, 0.1)),
            float(np.quantile(fold_errors, 0.9)),
            float(np.std(fold_errors, ddof=1)) if len(fold_errors) > 1 else 0.0,
        ]

        if error <= best_error:
            best_error = error
            lambda_opt = float(lam)
            beta_opt = beta_start.copy()
            higher_error_count = 0
        else:
            higher_error_count += 1

        if abs(error - previous_error) < opt_tol:
            early_stopping = i + 1
            break
        if higher_error_count >= early_stopping_max:
            early_stopping = i + 1
            break
        previous_error = error

    error_rows = error_rows[:early_stopping]
    beta_path = beta_path[:early_stopping]
    lambda_sd, beta_sd = _one_se_lambda(error_rows, beta_path, best_error, lambda_opt)
    data_beta = np.column_stack([error_rows[:, 0], beta_path])
    return CocoResult(
        lambda_opt=lambda_opt,
        lambda_sd=lambda_sd,
        beta_opt=beta_opt,
        beta_sd=beta_sd,
        data_error=error_rows,
        data_beta=data_beta,
        early_stopping=early_stopping,
        feature_names=feature_names,
        mean_z=mean_z,
        sd_z=sd_z,
        mean_y=mean_y,
        sd_y=sd_y,
        center_z=center_z,
        scale_z=scale_z,
        center_y=center_y,
        scale_y=scale_y,
    )


def lasso_covariance_block(
    x1: np.ndarray,
    z2: np.ndarray,
    y: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    lam: float,
    noise: Noise,
    ratio_matrix: np.ndarray | None,
    beta1_start: np.ndarray,
    beta2_start: np.ndarray,
    penalty: Penalty = "lasso",
    max_iter: int = 1000,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating block descent for BD-CoCoLasso."""
    n = y.shape[0]
    beta1 = beta1_start.copy()
    beta2 = beta2_start.copy()

    if noise == "missing":
        if ratio_matrix is None:
            raise ValueError("ratio_matrix is required for missing noise.")
        ratio_diag = np.diag(ratio_matrix)
        z2_tilde = z2 / ratio_diag
    else:
        ratio_diag = None
        z2_tilde = z2

    for _ in range(max_iter):
        beta1_old = beta1.copy()
        beta2_old = beta2.copy()

        rho1_partial = x1.T @ (y - z2_tilde @ beta2) / n
        beta1 = lasso_covariance(
            sigma1, rho1_partial, lam, beta1, penalty=penalty, opt_tol=opt_tol
        )

        residual = y - x1 @ beta1
        rho2_partial = z2.T @ residual / n
        if noise == "missing":
            rho2_partial = rho2_partial / ratio_diag
        beta2 = lasso_covariance(
            sigma2, rho2_partial, lam, beta2, penalty=penalty, opt_tol=opt_tol
        )

        if (
            np.sum(np.abs(beta1 - beta1_old)) < opt_tol
            and np.sum(np.abs(beta2 - beta2_old)) < opt_tol
        ):
            break

    beta1[np.abs(beta1) < zero_threshold] = 0.0
    beta2[np.abs(beta2) < zero_threshold] = 0.0
    return beta1, beta2


def _cv_matrices_block(
    z: np.ndarray,
    k: int,
    p1: int,
    p2: int,
    noise: Noise,
    tau: float | None,
    ratio_matrix: np.ndarray | None,
    mode: ProjectionMode,
    mu: float,
    etol: float,
    rng: np.random.Generator,
):
    n = z.shape[0]
    folds = _make_folds(n, k, rng)
    x1 = z[:, :p1]
    z2 = z[:, p1 : p1 + p2]

    sigma_global_uncorrupted = x1.T @ x1 / n
    sigma_global_corrupted = project_covariance(
        corrected_covariance(z2, noise, tau=tau, ratio_matrix=ratio_matrix),
        mode=mode,
        ratio=ratio_matrix,
        mu=mu,
        etol=etol,
    )

    sigma1_train = []
    sigma1_test = []
    sigma2_train = []
    sigma2_test = []
    for fold in range(k):
        test_idx = folds == fold
        train_idx = ~test_idx
        x1_train, x1_test = x1[train_idx], x1[test_idx]
        z2_train, z2_test = z2[train_idx], z2[test_idx]
        sigma1_train.append(x1_train.T @ x1_train / x1_train.shape[0])
        sigma1_test.append(x1_test.T @ x1_test / x1_test.shape[0])
        sigma2_train.append(
            project_covariance(
                corrected_covariance(z2_train, noise, tau=tau, ratio_matrix=ratio_matrix),
                mode=mode,
                ratio=ratio_matrix,
                mu=mu,
                etol=etol,
            )
        )
        sigma2_test.append(
            project_covariance(
                corrected_covariance(z2_test, noise, tau=tau, ratio_matrix=ratio_matrix),
                mode=mode,
                ratio=ratio_matrix,
                mu=mu,
                etol=etol,
            )
        )
    return (
        sigma_global_uncorrupted,
        sigma_global_corrupted,
        sigma1_train,
        sigma1_test,
        sigma2_train,
        sigma2_test,
        folds,
    )


def blockwise_coordinate_descent(
    z: np.ndarray,
    y: np.ndarray,
    *,
    p1: int,
    p2: int,
    center_z: bool = True,
    scale_z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: float | None = None,
    steps: int = 100,
    k: int = 4,
    mu: float = 10.0,
    tau: float | None = None,
    etol: float = 1e-4,
    opt_tol: float = 1e-5,
    early_stopping_max: int = 10,
    noise: Noise = "additive",
    penalty: Penalty = "lasso",
    mode: ProjectionMode = "ADMM",
    random_state: int | None = None,
    feature_names: list[str] | None = None,
) -> CocoResult:
    z, y = _as_2d_x_y(z, y)
    n, p = z.shape
    if p1 + p2 != p:
        raise ValueError("p1 + p2 must equal the number of columns in Z.")
    if lambda_factor is None:
        lambda_factor = 0.01 if n < p else 0.001
    if noise == "missing" and not center_z:
        raise ValueError("center_z=True is required for missing data.")

    z2_original = z[:, p1:p]
    ratio_matrix = _ratio_matrix_missing(z2_original) if noise == "missing" else None
    z_work, y_work, mean_z, sd_z, mean_y, sd_y = _preprocess(
        z, y, center_z, scale_z, center_y, scale_y
    )
    x1 = z_work[:, :p1]
    z2 = z_work[:, p1:p]

    rho1 = x1.T @ y_work / n
    rho2 = z2.T @ y_work / n
    if noise == "missing":
        rho2 = rho2 / np.diag(ratio_matrix)
    lambda_max = float(max(np.max(np.abs(rho1)), np.max(np.abs(rho2))))
    lambda_values = _logspace_desc(lambda_max, lambda_factor * lambda_max, steps)

    rng = np.random.default_rng(random_state)
    (
        sigma1_global,
        sigma2_global,
        sigma1_train,
        sigma1_test,
        sigma2_train,
        sigma2_test,
        folds,
    ) = _cv_matrices_block(
        z_work, k, p1, p2, noise, tau, ratio_matrix, mode, mu, etol, rng
    )

    beta1_start = np.zeros(p1)
    beta2_start = np.zeros(p2)
    beta_path = np.zeros((steps, p))
    error_rows = np.zeros((steps, 5))
    best_error = np.inf
    lambda_opt = lambda_values[0]
    beta_opt = np.zeros(p)
    early_stopping = steps
    higher_error_count = 0
    previous_error = np.inf

    for i, lam in enumerate(lambda_values):
        fold_errors = []
        for fold in range(k):
            test_idx = folds == fold
            train_idx = ~test_idx
            beta1_cv, beta2_cv = lasso_covariance_block(
                x1[train_idx],
                z2[train_idx],
                y_work[train_idx],
                sigma1_train[fold],
                sigma2_train[fold],
                lam,
                noise,
                ratio_matrix,
                beta1_start,
                beta2_start,
                penalty=penalty,
            )

            x1_test = x1[test_idx]
            z2_test = z2[test_idx]
            y_test = y_work[test_idx]
            n_test = y_test.shape[0]
            rho1_test = x1_test.T @ y_test / n_test
            if noise == "missing":
                ratio_diag = np.diag(ratio_matrix)
                z2_tilde_test = z2_test / ratio_diag
                rho2_test = z2_test.T @ y_test / n_test / ratio_diag
            else:
                z2_tilde_test = z2_test
                rho2_test = z2_test.T @ y_test / n_test
            cross = z2_tilde_test.T @ x1_test @ beta1_cv / n_test
            error = (
                beta1_cv @ sigma1_test[fold] @ beta1_cv
                + beta2_cv @ sigma2_test[fold] @ beta2_cv
                - 2.0 * rho1_test @ beta1_cv
                - 2.0 * rho2_test @ beta2_cv
                + 2.0 * beta2_cv @ cross
            )
            fold_errors.append(float(error))

        fold_errors = np.asarray(fold_errors)
        error = float(np.mean(fold_errors))
        beta1_start, beta2_start = lasso_covariance_block(
            x1,
            z2,
            y_work,
            sigma1_global,
            sigma2_global,
            lam,
            noise,
            ratio_matrix,
            beta1_start,
            beta2_start,
            penalty=penalty,
        )
        beta = np.concatenate([beta1_start, beta2_start])
        beta_path[i] = beta
        error_rows[i] = [
            lam,
            error,
            float(np.quantile(fold_errors, 0.1)),
            float(np.quantile(fold_errors, 0.9)),
            float(np.std(fold_errors, ddof=1)) if len(fold_errors) > 1 else 0.0,
        ]

        if error <= best_error:
            best_error = error
            lambda_opt = float(lam)
            beta_opt = beta.copy()
            higher_error_count = 0
        else:
            higher_error_count += 1

        if abs(error - previous_error) < opt_tol:
            early_stopping = i + 1
            break
        if i >= steps / 2 and error >= error_rows[0, 1]:
            early_stopping = i + 1
            break
        if higher_error_count >= early_stopping_max:
            early_stopping = i + 1
            break
        previous_error = error

    error_rows = error_rows[:early_stopping]
    beta_path = beta_path[:early_stopping]
    lambda_sd, beta_sd = _one_se_lambda(error_rows, beta_path, best_error, lambda_opt)
    data_beta = np.column_stack([error_rows[:, 0], beta_path])
    return CocoResult(
        lambda_opt=lambda_opt,
        lambda_sd=lambda_sd,
        beta_opt=beta_opt,
        beta_sd=beta_sd,
        data_error=error_rows,
        data_beta=data_beta,
        early_stopping=early_stopping,
        feature_names=feature_names,
        mean_z=mean_z,
        sd_z=sd_z,
        mean_y=mean_y,
        sd_y=sd_y,
        center_z=center_z,
        scale_z=scale_z,
        center_y=center_y,
        scale_y=scale_y,
    )


def coco(
    z: np.ndarray,
    y: np.ndarray,
    *,
    p1: int | None = None,
    p2: int | None = None,
    center_z: bool = True,
    scale_z: bool = True,
    center_y: bool = True,
    scale_y: bool = True,
    lambda_factor: float | None = None,
    steps: int = 100,
    k: int = 4,
    mu: float = 10.0,
    tau: float | None = None,
    etol: float = 1e-4,
    opt_tol: float = 1e-5,
    early_stopping_max: int = 10,
    noise: Noise = "additive",
    block: bool = True,
    penalty: Penalty = "lasso",
    mode: ProjectionMode = "ADMM",
    random_state: int | None = None,
    feature_names: list[str] | None = None,
) -> CocoResult:
    """Fit CoCoLasso or two-block BD-CoCoLasso."""
    if block:
        if p1 is None or p2 is None:
            raise ValueError("p1 and p2 are required when block=True.")
        return blockwise_coordinate_descent(
            z,
            y,
            p1=p1,
            p2=p2,
            center_z=center_z,
            scale_z=scale_z,
            center_y=center_y,
            scale_y=scale_y,
            lambda_factor=lambda_factor,
            steps=steps,
            k=k,
            mu=mu,
            tau=tau,
            etol=etol,
            opt_tol=opt_tol,
            early_stopping_max=early_stopping_max,
            noise=noise,
            penalty=penalty,
            mode=mode,
            random_state=random_state,
            feature_names=feature_names,
        )
    return pathwise_coordinate_descent(
        z,
        y,
        center_z=center_z,
        scale_z=scale_z,
        center_y=center_y,
        scale_y=scale_y,
        lambda_factor=lambda_factor,
        steps=steps,
        k=k,
        mu=mu,
        tau=tau,
        etol=etol,
        opt_tol=opt_tol,
        early_stopping_max=early_stopping_max,
        noise=noise,
        penalty=penalty,
        mode=mode,
        random_state=random_state,
        feature_names=feature_names,
    )


def _coefficients_from_result(result: CocoResult, beta: np.ndarray) -> tuple[np.ndarray, float]:
    coef = beta.copy()
    if result.scale_z:
        coef = coef / result.sd_z
    if result.scale_y:
        coef = coef * result.sd_y
    intercept = 0.0
    if result.center_y:
        intercept += result.mean_y
    if result.center_z:
        intercept -= float(result.mean_z @ coef)
    return coef, intercept


class CoCoLasso:
    """Class wrapper around the functional CoCoLasso implementation."""

    def __init__(
        self,
        *,
        p1: int | None = None,
        p2: int | None = None,
        center_z: bool = True,
        scale_z: bool = True,
        center_y: bool = True,
        scale_y: bool = True,
        lambda_factor: float | None = None,
        steps: int = 100,
        k: int = 4,
        mu: float = 10.0,
        tau: float | None = None,
        etol: float = 1e-4,
        opt_tol: float = 1e-5,
        early_stopping_max: int = 10,
        noise: Noise = "additive",
        block: bool = True,
        penalty: Penalty = "lasso",
        mode: ProjectionMode = "ADMM",
        random_state: int | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.center_z = center_z
        self.scale_z = scale_z
        self.center_y = center_y
        self.scale_y = scale_y
        self.lambda_factor = lambda_factor
        self.steps = steps
        self.k = k
        self.mu = mu
        self.tau = tau
        self.etol = etol
        self.opt_tol = opt_tol
        self.early_stopping_max = early_stopping_max
        self.noise = noise
        self.block = block
        self.penalty = penalty
        self.mode = mode
        self.random_state = random_state
        self.feature_names = feature_names

        self.result_: CocoResult | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coef_opt_: np.ndarray | None = None
        self.intercept_opt_: float | None = None
        self.lambda_opt_: float | None = None
        self.lambda_sd_: float | None = None
        self.data_error_: np.ndarray | None = None
        self.data_beta_: np.ndarray | None = None
        self.early_stopping_: int | None = None

    def fit(self, z: np.ndarray, y: np.ndarray) -> "CoCoLasso":
        result = coco(
            z,
            y,
            p1=self.p1,
            p2=self.p2,
            center_z=self.center_z,
            scale_z=self.scale_z,
            center_y=self.center_y,
            scale_y=self.scale_y,
            lambda_factor=self.lambda_factor,
            steps=self.steps,
            k=self.k,
            mu=self.mu,
            tau=self.tau,
            etol=self.etol,
            opt_tol=self.opt_tol,
            early_stopping_max=self.early_stopping_max,
            noise=self.noise,
            block=self.block,
            penalty=self.penalty,
            mode=self.mode,
            random_state=self.random_state,
            feature_names=self.feature_names,
        )
        self.result_ = result
        self.lambda_opt_ = result.lambda_opt
        self.lambda_sd_ = result.lambda_sd
        self.data_error_ = result.data_error
        self.data_beta_ = result.data_beta
        self.early_stopping_ = result.early_stopping

        self.coef_, self.intercept_ = _coefficients_from_result(result, result.beta_sd)
        self.coef_opt_, self.intercept_opt_ = _coefficients_from_result(result, result.beta_opt)
        return self

    def predict(self, newx: np.ndarray, lambda_pred: float | None = None) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.result_.predict(newx, lambda_pred=lambda_pred)


__all__ = [
    "CocoResult",
    "admm_proj",
    "hm_proj",
    "lasso_covariance",
    "lasso_covariance_block",
    "pathwise_coordinate_descent",
    "blockwise_coordinate_descent",
    "coco",
    "CoCoLasso",
]
