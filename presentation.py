import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # ax.scatter(CD[Ct_err>=0], SP[Ct_err>=0], Ct_err[Ct_err>=0], c = Ct_err[Ct_err>=0]/np.max(Ct_err[Ct_err>=0]), cmap = plt.cm.plasma_r, alpha = 0.6, label = 'error >= 0')
# ax.scatter(CD, SP, Ct_err, c = np.sign(Ct_err), cmap = plt.cm.plasma_r, alpha = 0.6,label = 'error < 0')
# ax.set_xlabel('width')
# ax.set_ylabel('Spacing')
# ax.set_zlabel('estimated adjustment')
# plt.show()
plt.clf()
fig, ax = plt.subplots()
ax.scatter(train_X[:,0], train_X[:,1], s = abs(train_Y)*20, c = np.sign(train_Y), cmap = plt.cm.plasma_r, alpha=0.5)
ax.set_xlabel('width')
ax.set_ylabel('Spacing')
plt.legend()
plt.show()