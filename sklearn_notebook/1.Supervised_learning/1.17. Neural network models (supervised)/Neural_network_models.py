# 1.17.1. Multi-layer Perceptron
# 1.17.2. Classification
"""
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)


print(clf.predict([[2., 2.], [-1., -2.]]))
print([coef.shape for coef in clf.coefs_])
print(clf.predict_proba([[2., 2.], [1., 2.]]) )
"""
# 1.17.3. Regression

# 1.17.4. Regularization

# 1.17.5. Algorithms

# 1.17.6. Complexity

# 1.17.7. Mathematical formulation




