from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True)
print(data)
print(target.shape)