import numpy as np
import matplotlib.pyplot as plt

# Generating 100 i.i.d random sampled 4x4 matrices of noise
samples = np.random.randn(100, 8, 8)

new_samples = np.random.randn(100, 8, 8)


# Reshaping the samples to a 2D array where each row is a flattened 4x4 matrix
reshaped_samples = samples.reshape(100, -1)

n_reshaped_samples = new_samples.reshape(100, -1)


# Computing the covariance matrix of the reshaped samples
covariance_matrix = np.cov(reshaped_samples, rowvar=False)

# import pdb; pdb.set_trace()

# covariance_matrix = np.random.rand(64,64)

# Visualizing the 16x16 covariance matrix
plt.figure(figsize=(10, 8))
plt.imshow(covariance_matrix, cmap='viridis')
plt.title('16x16 Covariance Matrix of 4x4 Noise Samples')
plt.colorbar()
plt.show()
plt.savefig('/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst')