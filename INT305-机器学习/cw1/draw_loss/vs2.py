# Add hyperparameter effects to the two 2D visualizations.
# - For SVM: overlay decision boundaries for multiple L2 regularization strengths (reg λ).
#   Also annotate each with geometric margin 1/||w||.
# - For Softmax (logistic): overlay boundaries for multiple L2 regularization strengths.
#   Keep one probability heatmap (using a middle reg) so the plot remains readable.

import numpy as np
import matplotlib.pyplot as plt

# Recreate the same dataset for reproducibility
rs = np.random.RandomState(7)
n_per = 120
mean_pos = np.array([2.0, 2.0])
mean_neg = np.array([-1.5, -1.0])
cov_pos = np.array([[0.8, 0.2],[0.2, 0.6]])
cov_neg = np.array([[0.7, -0.15],[-0.15, 0.5]])
X_pos = rs.multivariate_normal(mean_pos, cov_pos, size=n_per)
X_neg = rs.multivariate_normal(mean_neg, cov_neg, size=n_per)
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(n_per), -np.ones(n_per)])  # +1 / -1

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0],1))])

def train_linear_svm(X, y, lr=0.05, reg=1e-2, epochs=200):
    Xb = add_bias(X)
    theta = np.zeros(Xb.shape[1])
    for ep in range(epochs):
        idx = np.arange(X.shape[0]); np.random.shuffle(idx)
        for i in idx:
            xi = Xb[i]; yi = y[i]
            margin = yi * np.dot(theta, xi)
            if margin < 1.0:
                grad = -yi * xi + reg * np.r_[theta[:-1], 0.0]
            else:
                grad = reg * np.r_[theta[:-1], 0.0]
            theta -= lr * grad
    return theta

def train_logreg(X, y, lr=0.05, reg=1e-2, epochs=200):
    y01 = (y == 1).astype(float)
    Xb = add_bias(X)
    theta = np.zeros(Xb.shape[1])
    for ep in range(epochs):
        idx = np.arange(X.shape[0]); np.random.shuffle(idx)
        for i in idx:
            xi = Xb[i]; yi = y01[i]
            z = np.dot(theta, xi)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (p - yi) * xi + reg * np.r_[theta[:-1], 0.0]
            theta -= lr * grad
    return theta

def plot_data(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X_pos[:,0], X_pos[:,1], s=24, alpha=0.9, label="+1 class")
    ax.scatter(X_neg[:,0], X_neg[:,1], s=24, alpha=0.9, label="-1 class")
    ax.legend(loc="upper left")

def line_from_theta(theta, level=0.0, xlim=None):
    w = theta[:2]; b = theta[2]
    if xlim is None:
        xlim = plt.gca().get_xlim()
    xs = np.linspace(xlim[0], xlim[1], 400)
    if abs(w[1]) < 1e-10:
        ys = np.full_like(xs, fill_value=np.nan)
    else:
        ys = (level - w[0]*xs - b)/w[1]
    return xs, ys

# Axis ranges
xlim = (X[:,0].min()-1.5, X[:,0].max()+1.5)
ylim = (X[:,1].min()-1.5, X[:,1].max()+1.5)

# --- Figure 1: SVM with multiple λ ---
regs_svm = [5e-3, 1e-2, 5e-2, 2e-1]  # smaller -> weaker regularization (larger C)
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

plt.figure(figsize=(7,5))
plot_data()
ax = plt.gca()
ax.set_xlim(xlim); ax.set_ylim(ylim)
for reg, c in zip(regs_svm, colors):
    theta = train_linear_svm(X, y, lr=0.05, reg=reg, epochs=250)
    w = theta[:2]; margin = 1.0 / (np.linalg.norm(w) + 1e-12)
    xs, ys = line_from_theta(theta, level=0.0, xlim=xlim)
    ax.plot(xs, ys, color=c, linewidth=2, label=f"SVM λ={reg:g}, geom. margin≈{margin:.2f}")
# Draw one pair of margin bands for the middle reg to avoid clutter
theta_mid = train_linear_svm(X, y, lr=0.05, reg=regs_svm[1], epochs=250)
xs, ys_pos = line_from_theta(theta_mid, level=+1.0, xlim=xlim)
xs, ys_neg = line_from_theta(theta_mid, level=-1.0, xlim=xlim)
ax.plot(xs, ys_pos, linestyle="--", linewidth=1, color="#999999", label="margin bands (example)")
ax.plot(xs, ys_neg, linestyle="--", linewidth=1, color="#999999")
ax.set_title("Linear SVM: effect of L2 regularization λ on boundary\n(larger margin for smaller ||w||)")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.grid(True)
ax.legend(loc="lower right", fontsize=9)
plt.show()

# --- Figure 2: Softmax/logistic with multiple λ ---
regs_log = [5e-3, 1e-2, 5e-2, 2e-1]
plt.figure(figsize=(7,5))
# Base probability heatmap using a mid reg for readability
theta_base = train_logreg(X, y, lr=0.05, reg=regs_log[1], epochs=250)
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300),
                     np.linspace(ylim[0], ylim[1], 300))
grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
zz = grid @ theta_base
pp = 1.0 / (1.0 + np.exp(-zz))
cs = plt.contourf(xx, yy, pp.reshape(xx.shape), levels=20, alpha=0.7)
plt.colorbar(cs, label="P(y=+1)  (base λ)")
plot_data()
for reg, c in zip(regs_log, colors):
    theta = train_logreg(X, y, lr=0.05, reg=reg, epochs=250)
    xs, ys = line_from_theta(theta, level=0.0, xlim=xlim)  # p=0.5 boundary
    plt.plot(xs, ys, color=c, linewidth=2, label=f"Logistic λ={reg:g}")
plt.xlim(xlim); plt.ylim(ylim)
plt.title("Softmax: effect of L2 regularization λ on boundary\n(probability field shown for a base λ)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.grid(True)
plt.legend(loc="lower right", fontsize=9)
plt.show()
