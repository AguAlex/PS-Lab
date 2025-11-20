import numpy as np
import scipy
from matplotlib import pyplot as plt


np.random.seed(0)

cmap = plt.colormaps["gray"]

raton = scipy.datasets.face(gray=True)

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=np.shape(raton))
raton_noise = raton + noise

plt.imshow(raton_noise, cmap=cmap)
plt.show()