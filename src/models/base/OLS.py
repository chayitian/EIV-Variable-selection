import numpy as np
from sklearn.preprocessing import StandardScaler


class OLS:
    """
    普通最小二乘回归（Ordinary Least Squares）

    目标函数：
    (1 / (2n)) * ||y - Xβ||^2

    参数
    ----------
    fit_intercept : bool
        是否拟合截距项
    normalize : bool
        是否对特征进行标准化
    """

    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        self.coef_ = None
        self.intercept_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def fit(self, X, y):
        """
        拟合OLS

        参数
        ----------
        X : np.ndarray
            协变量矩阵 (n_samples, n_features)
        y : np.ndarray
            响应变量向量 (n_samples,)
        """
        n_samples, n_features = X.shape

        if self.normalize:
            self.scaler_X_ = StandardScaler()
            self.scaler_y_ = StandardScaler(with_std=False)

            X_scaled = self.scaler_X_.fit_transform(X)
            y_centered = self.scaler_y_.fit_transform(y.reshape(-1, 1)).flatten()

            X_std = self.scaler_X_.scale_

            Sigma_X = (X_scaled.T @ X_scaled) / n_samples
            min_eig = np.min(np.linalg.eigvalsh(Sigma_X))
            if min_eig < 1e-10:
                Sigma_X += np.eye(n_features) * (1e-10 - min_eig)

            rho = (X_scaled.T @ y_centered) / n_samples

            beta_scaled = np.linalg.solve(Sigma_X, rho)

            beta_original_scale = beta_scaled / X_std
            self.coef_ = beta_original_scale
            self.intercept_ = self.scaler_y_.mean_ - np.dot(self.scaler_X_.mean_, beta_original_scale)
        else:
            if self.fit_intercept:
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)

                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_mean = np.zeros(n_features)
                y_mean = 0.0

                X_centered = X
                y_centered = y

            Sigma_X = (X_centered.T @ X_centered) / n_samples
            min_eig = np.min(np.linalg.eigvalsh(Sigma_X))
            if min_eig < 1e-10:
                Sigma_X += np.eye(n_features) * (1e-10 - min_eig)

            rho = (X_centered.T @ y_centered) / n_samples

            self.coef_ = np.linalg.solve(Sigma_X, rho)

            if self.fit_intercept:
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
            else:
                self.intercept_ = 0.0

        return self

    def predict(self, X):
        """
        预测

        参数
        ----------
        X : np.ndarray
            协变量矩阵 (n_samples, n_features)

        返回
        -------
        y_pred : np.ndarray
            预测的响应变量 (n_samples,)
        """
        return X @ self.coef_ + self.intercept_
