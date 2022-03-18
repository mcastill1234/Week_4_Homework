import numpy as np
import HyperplaneProcedures as hp

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

# Problems 1A and 1B:
dim, num_samples = data.shape
red_margins = np.zeros((1, num_samples))
blue_margins = np.zeros((1, num_samples))
for i in range(num_samples):
    red_margins[0, i] = labels[0, i] * hp.signed_dist(data[:, i], red_th, red_th0)
    blue_margins[0, i] = labels[0, i] * hp.signed_dist(data[:, i], blue_th, blue_th0)
red_s_sum = np.sum(red_margins)
blue_s_sum = np.sum(blue_margins)
red_s_min = red_margins.min()
blue_s_min = blue_margins.min()
red_s_max = red_margins.max()
blue_s_smax = blue_margins.max()
answer1a = [red_s_sum, red_s_min, red_s_max]
answer1b = [blue_s_sum, blue_s_min, blue_s_smax]
print("The values of each score (Ssum, Smin, Smax) on the red separator are: ", answer1a)
print("The values of each score (Ssum, Smin, Smax) on the blue separator are: ", answer1b)




