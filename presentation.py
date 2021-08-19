import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import random
df = pd.read_csv('QAResultData.csv')
data = np.array(df.values)
df_keys = np.array([[eval(i)[0], eval(i)[1]] for i in data[:,0]])
logi_pool = [(file_name[0][:2] == 'M'+str(1))&(file_name[0][2] == 'M')&('SRM' not in file_name[0]) for file_name in df_keys] 
parameter = df_keys[logi_pool][:,1]
Ct_err = data[logi_pool][:,2]
Cc_err = data[logi_pool][:,3]
CD = data[logi_pool][:,4]   
SP = data[logi_pool][:,5]
train_X = np.array([[CD[i], SP[i]] for i in range(0, len(CD), 2)])
train_Y = np.array([(Ct_err[i]*parameter[i+1][0]-Ct_err[i+1]*parameter[i][0])/(Ct_err[i]-Ct_err[i+1]) for i in range(0, len(Ct_err), 2)])

from sklearn.model_selection import train_test_split
regressor = DecisionTreeRegressor(random_state=0, min_samples_leaf = 5)
regressor.fit(train_X, train_Y)
# X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.20,
#                                                     random_state=0,
#                                                     shuffle=True
#                                                     )
# from sklearn.model_selection import cross_val_score
# regressor = DecisionTreeRegressor(min_samples_leaf = 40,max_leaf_nodes = 90,  criterion = 'mse', random_state=0)
# regressor.fit(X_train, y_train)
# print(regressor.score(X_train, y_train))
# print(regressor.score(X_test, y_test))
# ax = fig.gca(projection='3d')
# # ax.scatter(CD[Ct_err>=0], SP[Ct_err>=0], Ct_err[Ct_err>=0], c = Ct_err[Ct_err>=0]/np.max(Ct_err[Ct_err>=0]), cmap = plt.cm.plasma_r, alpha = 0.6, label = 'error >= 0')
# ax.scatter(CD, SP, Ct_err, c = np.sign(Ct_err), cmap = 'PRGn', alpha = 0.6,label = 'error < 0')
# ax.set_xlabel('width')
# ax.set_ylabel('Spacing')
# ax.set_zlabel('estimated adjustment')
# plt.show()
# plt.clf()
# fig, ax = plt.subplots()
# ax.scatter(train_X[:,0], train_X[:,1], s = [abs(Ct_err[i])*60 for i in range(0, len(Ct_err), 2)], c = [np.sign(Ct_err[i])*20 for i in range(0, len(Ct_err), 2)], cmap = 'PRGn', alpha=0.5)
# ax.set_xlabel('width')
# ax.set_ylabel('Spacing')
# plt.legend()
# plt.show()

# X = np.linspace(0, 1, 100)
# Y = []
# for i in X:
#     if i < 0.3:
#         Y.append(2+random.randint(-10, 10)/100)
#     elif i<0.6:
#         Y.append(0+random.randint(-10, 10)/100)
#     else:
#         Y.append(1+random.randint(-10, 10)/100)
# # plt.scatter(X, Y)
# # plt.show()
# mse = []
# for i in range(0,len(Y)):
#     mse.append(np.sum((np.mean(np.array(Y)[:i])-np.array(Y)[:i])**2)/len(np.array(Y)[:i]))
# plt.xlabel('data')
# plt.ylabel('Mean square error')
# plt.plot(X, mse)
# plt.show()

x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(100, 650, 1000),
                     np.linspace(100, 650, 1000))

Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn, levels = 20)
plt.colorbar()
plt.scatter(train_X[:,0], train_X[:,1], s = abs(train_Y)*10, c = np.sign(train_Y), cmap = 'plasma', alpha=0.5)

plt.show()