from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


img = np.array(Image.open('pic01.png'))
print(img.shape)

img = img.reshape(-1, 4)

label_model = KMeans(n_clusters = 3)
label_model.fit(img)

label_class = label_model.predict(img)

plt.imshow(label_class.reshape(2227, 2393, 4))
plt.show()