import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

x,y = fetch_olivetti_faces(return_X_y=True)
print(x.shape)
