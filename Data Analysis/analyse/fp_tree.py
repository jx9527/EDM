import pandas as pd
sysboml = pd.read_csv('sysboml.csv')
from fpgrowth_py import fpgrowth
sysboml = sysboml.drop(['Unnamed: 0', 'ID'], axis=1)
# sysboml = sysboml.values.tolist()
# freqItemSet, rules = fpgrowth(sysboml, minSupRatio=0.4, minConf=0.2)
#
# for i in range(len(rules)):
#     if  len(rules[i][1])==1:
#         print(rules[i])
#     if  len(rules[i][1])==1 and 'A2' in rules[i][1]:
    #     print(rules[i])