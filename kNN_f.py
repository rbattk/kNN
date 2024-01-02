# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:32:38 2021

@author: user1
"""
import numpy as np
import pandas as pd 
from math import sqrt
#-----------------------------------------------------------------------------#
#--Step 1: Upload data--------------------------------------------------------#
path = "C:\\Users\\rabia\\Desktop\\Published\\seeds_dataset.txt"
seed_data = np.loadtxt (path)#,names=['area','perimete','compactnes', 'length of kerne','width of kernel','asymmetry coefficient' , 'length of kernel groov', 'class']) 
sd = pd.DataFrame(seed_data, columns=(['area','perimeter','compactnes', 
                                       'length of kerne','width of kernel',
                                       'asymmetry coefficient' , 
                                       'length of kernel groov', 'Class'])) 
#-----------------------------------------------------------------------------#
#--Step 2 Separate features and classes---------------------------------------#
features = ['area','perimeter','compactnes', 'length of kerne',
            'width of kernel','asymmetry coefficient' , 
            'length of kernel groov'] 
x = sd.loc[:, features].values #loc=for specific row/column access
c =  sd.loc[:,['Class']].values 

#-----------------------------------------------------------------------------#
#--Step 3 Calculating euclidean distance--------------------------------------#
K=9;
t = [];
u = [];
ss = [];
c_new = [];
p = [];
c_mat = np.zeros((3,3));
for z in range(210):    
    for j in range(209): 
        for i in range(7):              
            X = [x[j][:][i]]
            Y = [x[z][:][i]]
            def öklid(X,Y) :
                temp=0;
                for x,y in zip(X,Y):
                    temp += pow(x-y,2)        
                return((temp))
            k =  öklid(X,Y);
            t.append(k);
            h = sqrt(sum(t))    
        #print('Euclidean Distance:',h)
#-----------------------------------------------------------------------------#
#--Step 4 k-NN----------------------------------------------------------------#        
        u.append(h);
        t.clear();  
    l = min(u);
    b = u.index(l)    
    smin = np.sort(u);
    smin = np.delete(smin,0,0)
    n=smin[0:K]
    rt = [0,0,0];
    for h in range(K):
        ss.append(u.index(n[h]));
        if ss[h]<70:
            rt[0] = rt[0]+1;
        elif 70<=ss[h]<140:
            rt[1] = rt[1]+1;
        else:
            rt[2] = rt[2]+1;     
    
    c_new.append(rt.index(max(rt))+1)
    #print('Indices of closest feature vectors:',ss)
    u.clear();
    ss.clear();
print('The classification result:',c_new);    
#-----------------------------------------------------------------------------#
#--Step 5: Obtaining classification success criteria--------------------------#
#-----------------------------------------------------------------------------#        
for t in range(70):
    if c_new[t] == 1:
        c_mat[0,0] = c_mat[0,0] + 1;
    elif c_new[t] == 2:
        c_mat[0,1] = c_mat[0,1] + 1;
    elif c_new[t] == 3:
        c_mat[0,2] = c_mat[0,2] + 1;

for t in range(70):
    if c_new[t+70] == 1:
        c_mat[1,0] = c_mat[1,0] + 1;
    elif c_new[t+70] == 2:
        c_mat[1,1] = c_mat[1,1] + 1;
    elif c_new[t+70] == 3:
        c_mat[1,2] = c_mat[1,2] + 1;    
        
for t in range(70):
    if c_new[t+140] == 1:
        c_mat[2,0] = c_mat[2,0] + 1;
    elif c_new[t+140] == 2:
        c_mat[2,1] = c_mat[2,1] + 1;
    elif c_new[t+140] == 3:
        c_mat[2,2] = c_mat[2,2] + 1;     


cd = c_mat.diagonal();
Pc1 = cd[0] / sum(c_mat[:,0]);
Pc2 = cd[1] / sum(c_mat[:,1]);
Pc3 = cd[2] / sum(c_mat[:,2]);
Sn1 = cd[0] / sum(c_mat[0,:]);
Sn2 = cd[1] / sum(c_mat[1,:]);
Sn3 = cd[2] / sum(c_mat[2,:]);
F1C1 = 2*Pc1*Sn1 / (Pc1 + Sn1);
F1C2 = 2*Pc2*Sn2 / (Pc2 + Sn2);
F1C3 = 2*Pc3*Sn3 / (Pc3 + Sn3);
Sp1 = sum(sum(c_mat[1:3,1:3])) / (sum(sum(c_mat[1:3,1:3])) + sum(c_mat[1:3,0]));
Sp2 = ((c_mat[0,0] + c_mat[0,2] + c_mat[2,0] + c_mat[2,2])) / (((c_mat[0,0] + c_mat[0,2] + c_mat[2,0] + c_mat[2,2]) + c_mat[0,1] + c_mat[2,1]));
Sp3 = sum(sum(c_mat[0:2,0:2])) / (sum(sum(c_mat[0:2,0:2])) + sum(c_mat[0:2,2]));

Accuracy = c_mat.trace() / sum(sum(c_mat));
Specificity = (Sp1 + Sp2 +Sp3)/3;
Precision = (Pc1 + Pc2 + Pc3)/3;
Sensitivity = (Sn1 + Sn2 + Sn3)/3;
F1_Score = (F1C1 + F1C2 + F1C3)/3;
print('\n','Accuracy:',Accuracy,'\n','Sensitivity:',Sensitivity,
      '\n','Specificity:',Specificity,'\n','Precision:',Precision,
      '\n','F1_Score:',F1_Score);  
#-----------------------------------------------------------------------------#