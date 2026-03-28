import numpy as np


class LassoRegression:
    """
    Lasso 回归（坐标下降法实现）

    目标函数：
    (1 / (2n)) * ||y - Xβ||^2 + alpha * ||β||_1

    参数
    ----------
    alpha : float
        L1 正则化强度
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, rho, lam):
        """
        软阈值函数
        """
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    def fit(self, X, y):
        """
        使用坐标下降法拟合 Lasso
        """
        n_samples, n_features = X.shape

        # 计算均值
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)

        # 中心化
        X_centered = X - X_mean
        y_centered = y - y_mean

        # 初始化系数
        beta = np.zeros(n_features)

        for iteration in range(self.max_iter):

            beta_old = beta.copy()

            for j in range(n_features):

                # 计算残差（排除当前特征）
                residual = y_centered - X_centered @ beta + beta[j] * X_centered[:, j]

                # rho
                rho = np.dot(X_centered[:, j], residual) / n_samples

                # z
                z = np.sum(X_centered[:, j] ** 2) / n_samples

                # 坐标更新
                z_safe = max(z, 1e-12) ## 防止常数列或近常数列导致除零
                beta[j] = self._soft_threshold(rho, self.alpha) / z_safe

            # 收敛判断
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        # 保存结果
        self.coef_ = beta

        # 恢复截距
        self.intercept_ = y_mean - X_mean @ beta

        return self

    def predict(self, X):
        """
        预测
        """
        return X @ self.coef_ + self.intercept_