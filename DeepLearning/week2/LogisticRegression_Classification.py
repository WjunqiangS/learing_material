from planar_utils import load_planar_dataset, plot_decision_boundary
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import sklearn
import numpy as np

# X维度为2 * 400，第一行表是x的坐标，第二行代表是y的坐标
# Y维度为1 * 400，对应X的坐标
X, Y = load_planar_dataset()

# 根据指定坐标画出对应点；第一、二个参数为x、y的坐标，c:颜色，s:大小
plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0,:].shape), s=40, cmap=plt.cm.Spectral)
plt.show()

X_shape = X.shape
Y_shape = Y.shape

m = X_shape[1]

print('the shape of X is ' + str(X_shape))
print('the shape of Y is ' + str(Y_shape))
print('We have '+ str(m) + ' samples to train...')

# 获取逻辑回归模型
clf = sklearn.linear_model.LogisticRegressionCV();

# 根据训练样本训练逻辑回归模型
clf.fit(X.T, Y.reshape((Y_shape[1])).T);

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")