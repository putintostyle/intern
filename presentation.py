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

# from sklearn.model_selection import train_test_split
# regressor = DecisionTreeRegressor(random_state=0, min_samples_leaf = 5)
# regressor.fit(train_X, train_Y)
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
cs = np.array([(Ct_err[i])*60 for i in range(0, len(Ct_err), 2)])
fig, ax = plt.subplots()
ax.scatter(train_X[:,0][cs>0], train_X[:,1][cs>0], s = [abs(i) for i in cs[cs>0]], c = [np.sign(i)*20 for i in cs[cs>0]], cmap = 'PRGn_r', alpha=0.5, label = 'error < 0')
ax.scatter(train_X[:,0][cs<=0], train_X[:,1][cs<=0], s = [abs(i) for i in cs[cs<=0]], c = [np.sign(i)*20 for i in cs[cs<=0]], cmap = 'PRGn', alpha=0.5, label = 'error >= 0')
ax.set_xlabel('width')
ax.set_ylabel('Spacing')

plt.legend()
plt.show()
"""
X = np.linspace(0, 1, 100)
Y = []
for i in X:
    if i < 0.3:
        Y.append(2+random.randint(-10, 10)/100)
    elif i<0.6:
        Y.append(0+random.randint(-10, 10)/100)
    else:
        Y.append(1+random.randint(-10, 10)/100)
plt.scatter(X, Y)
plt.savefig('data.png')
mse = []

for i in range(0,len(Y)):
    tmp_1 = np.sum((np.mean(np.array(Y)[i:])-np.array(Y)[i:])**2)/len(np.array(Y)[i:])

    tmp_2 = np.sum((np.mean(np.array(Y)[:i])-np.array(Y)[:i])**2)/len(np.array(Y)[:i])
    if abs(tmp_1 - tmp_2)< 0.001:
        mean = [np.mean(np.array(Y)[i:]), np.mean(np.array(Y)[:i]) ]
        print(mean)
    mse.append([tmp_1, tmp_2])
mse = np.array(mse)

# plt.xlabel('data')
# plt.ylabel('Mean square error')
# plt.plot(X, mse[:,0], 'r', label = 'region 1')
# plt.plot(X, mse[:,1], 'b', label = 'region 2')
# plt.legend()
# plt.savefig('data with mse.png')
plt.clf()
x = np.linspace(0, 1, 1000)
y_1 = np.exp(-(x-0.2)**2/0.01)
y_2 = np.exp(-(x-0.5)**2/0.1)
plt.plot(x, y_1, 'b--', label="$y_1$ = $d_i$=0.2 $\epsilon$=0.01")
plt.plot(x, y_2, 'r--', label="$y_2$ = $d_i$=0.5 $\epsilon$=0.1")
plt.plot(x, 0.5*y_1+0.2*y_2, 'green', label = "$0.5y_1+0.2y_2$")
plt.legend()
plt.show()
# x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
# y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1

# xx, yy = np.meshgrid(np.linspace(100, 650, 1000),
                    #  np.linspace(100, 650, 1000))

# Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn, levels = 20)
# plt.colorbar()

# # plt.show()
# from scipy.interpolate import LinearNDInterpolator
# interp = LinearNDInterpolator(train_X, train_Y, fill_value=0)
# Z = interp(xx, yy)
# plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn, levels = 20)
# plt.colorbar()
# plt.scatter(train_X[:,0], train_X[:,1], s = abs(train_Y)*10, c = np.sign(train_Y), cmap = 'plasma', alpha=0.5)
# plt.xlabel('Width')
# plt.ylabel('Spacing')
# plt.show()
"""