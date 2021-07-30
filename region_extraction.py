import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import sys

if len(sys.argv) == 2:
    TREE_MIN = 20
elif len(sys.argv) == 3:
    TREE_MIN = sys.argv[2]

WORK_FOLDER = sys.argv[1]
DATA_FOLDER = '__tuning__'
PATH = os.path.join(WORK_FOLDER,DATA_FOLDER)

df = pd.read_csv(os.path.join(PATH, "QAResultData.csv"))
data = np.array(df.values)
df_keys = np.array([[eval(i)[0], eval(i)[1]] for i in data[:,0]])

 
logi_pool = [(file_name[0][:2] == 'M'+str(1))&(file_name[0][2] == 'M')&('SRM' not in file_name[0]) for file_name in df_keys] 

parameter = df_keys[logi_pool][:,1]
Ct_err = data[logi_pool][:,2]
Cc_err = data[logi_pool][:,3]
CD = data[logi_pool][:,4]   
SP = data[logi_pool][:,5]

train_X = np.array([[CD[i], SP[i]] for i in range(0, len(CD), 2)])
#(Ct_err1*wext2 - Ct_err2*wext1)/(Ct_err1 - Ct_err2)
train_Y = np.array([(Ct_err[i]*parameter[i+1][0]-Ct_err[i+1]*parameter[i][0])/(Ct_err[i]-Ct_err[i+1]) for i in range(0, len(Ct_err), 2)])

regressor = DecisionTreeRegressor(random_state=0, min_samples_leaf = TREE_MIN)
regressor.fit(train_X, train_Y)
def split(array, dim, thred):
    if dim == 0:
        return [[np.min(array[0]), thred], array[1]],  [[thred, np.max(array[0])], array[1]]
    elif dim == 1:
        return [array[0], [np.min(array[1]), thred]],  [array[0], [thred, np.max(array[1])]]

interval = [[0,1000], [0,1000]]
left_children = [interval]
right_children = [interval]
right_remove_idx = False
left_remove_idx = False
parent = []
region = []
LEFT_FLAG = False

for index_of_node in range(0,len(regressor.tree_.threshold)):
    if regressor.tree_.threshold[index_of_node] != -2:
        
#         print(right_children)
        
        if LEFT_FLAG == False:
            
            FROM_LEFT_FLAG = False
            
            right, left = split(right_children[-1], regressor.tree_.feature[index_of_node], regressor.tree_.threshold[index_of_node])
            
            parent.append([0,1,len(right_children)-1, False, False])
            right_children.append(right)
            left_children.append(left)
#             print(left_children)
            FROM_RIGHT_FLAG = True
        else:
            FROM_RIGHT_FLAG = False
            
            right, left = split(left_children[-1], regressor.tree_.feature[index_of_node], regressor.tree_.threshold[index_of_node])
            
            
            
            parent.append([1, 0,len(left_children)-1, False, False])
            
            right_children.append(right)
            left_children.append(left)
            LEFT_FLAG = False
            FROM_LEFT_FLAG = True
        
    else:
        
        
        if regressor.tree_.threshold[index_of_node-1] != -2:
            sub_region = right_children[-1]
            region.append([sub_region[0][0], sub_region[0][1], sub_region[1][0], sub_region[1][1], regressor.tree_.value[index_of_node][0][0]])
            
            parent_idx = parent[-1][0:3]
#             parent_remove = parent[-1][3:]
            
            right_children.pop()
            LEFT_FLAG = True
#             parent_remove[1] = True
            parent[-1][4] = True
        else:
            sub_region = left_children[-1]
            region.append([sub_region[0][0], sub_region[0][1], sub_region[1][0], sub_region[1][1], regressor.tree_.value[index_of_node][0][0]])
            
            parent_idx = parent[-1][0:3]
#             parent_remove = parent[-1][3:]

            left_children.pop()
            parent[-1][3] = True
     
    
#         print(left_remove_idx, right_remove_idx)
        while (parent[-1][3:][0] == True) & (parent[-1][3:][1] == True):
#             if len(parent == len(regressor.tree_.threshold)-1:
#                 break:
            
            if parent_idx[0] == 1: #left
                left_children.pop(parent_idx[2])
                parent.pop()
                if (index_of_node == len(regressor.tree_.feature)-1) & (len(parent) == 0):
                    break
                else:
                    parent_idx = parent[-1][0:3]
#                 parent_remove = parent[-1][3:]
                
                    parent[-1][3] = True
            else:
                right_children.pop(parent_idx[2])
                parent.pop()
                if (index_of_node == len(regressor.tree_.feature)-1) & (len(parent) == 0):
                    break
                else:
                    parent_idx = parent[-1][0:3]
    #                 parent_remove = parent[-1][3:]

                    parent[-1][4] = True

        else:
            continue
header = [['CD1', 'CD2', 'SP1', 'SP2', 'wext']]
import csv

with open('rules.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(header)
    writer.writerows(region)