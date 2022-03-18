import numpy as np
import HyperplaneProcedures as hp

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4

dim, num_samples = data.shape
hinge_loss = np.zeros((1, num_samples))
margin_ref = (2**0.5)/2
for i in range(num_samples):
    step_margin = labels[0, i] * hp.signed_dist(data[:, i], th, th0)
    if step_margin < margin_ref:
        hinge_loss[0, i] = 1 - step_margin/margin_ref
    else:
        hinge_loss[0, i] = 0
print(hinge_loss[0].tolist())

