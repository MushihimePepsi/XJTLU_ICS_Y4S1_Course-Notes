# Softmax (cross-entropy) vs SVM (hinge) on a common x-axis: relative margin z = s_wrong - s_correct.
# We plot:
#   L_ce(z)   = log(1 + exp(z))            (single rival; multiclass extends with log-sum-exp)
#   L_hinge(z)= max(0, z + Delta)          (margin hyperparameter Delta, typically 1)
# And their derivatives wrt z:
#   dL_ce/dz     = 1 / (1 + exp(-z))       (sigmoid)
#   dL_hinge/dz  = 1{z + Delta > 0}

import numpy as np
import matplotlib.pyplot as plt

# Define range and margin
z = np.linspace(-5, 5, 1000)
Delta = 1.0

# Losses
L_ce = np.log1p(np.exp(z))              # numerically stable log(1+exp(z))
L_hinge = np.maximum(0.0, z + Delta)

# Derivatives
sigmoid = 1.0 / (1.0 + np.exp(-z))
dL_ce = sigmoid
dL_hinge = (z + Delta > 0).astype(float)

# Plot losses
plt.figure(figsize=(7, 5))
plt.plot(z, L_ce, label="Softmax CE: log(1+exp(z))")
plt.plot(z, L_hinge, label=f"SVM hinge: max(0, z+{Delta:.0f})")
plt.axvline(x=-Delta, linestyle="--", linewidth=1)
plt.text(-Delta, 0.2, "z = -Δ", ha="right", va="bottom")
plt.xlabel("Relative margin  z = s_wrong - s_correct")
plt.ylabel("Loss")
plt.title("Loss vs. relative margin (single rival)")
plt.legend()
plt.grid(True)
plt.show()

# Plot derivatives
plt.figure(figsize=(7, 5))
plt.plot(z, dL_ce, label="d/dz Softmax CE = sigmoid(z)")
plt.plot(z, dL_hinge, label="d/dz Hinge = 1{z+Δ>0}")
plt.axvline(x=-Delta, linestyle="--", linewidth=1)
plt.text(-Delta, 0.05, "z = -Δ", ha="right", va="bottom")
plt.xlabel("Relative margin  z = s_wrong - s_correct")
plt.ylabel("Derivative to z")
plt.title("Gradient vs. relative margin (single rival)")
plt.legend()
plt.grid(True)
plt.show()
