import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


art = pd.read_csv('art.csv').sort_values(['student_id'])
sport = pd.read_csv('sport.csv').sort_values(['student_id'])
political = pd.read_csv('political.csv').sort_values(['student_id'])
subject = pd.read_csv('subject.csv')
diligence = pd.read_csv('diligence.csv').sort_values(['student_id'])
consumption = pd.read_csv('consumption.csv').sort_values(['student_id'])
#
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

# 分析知识水平的异常点
# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# plt.boxplot(x=[subject['subject_ave'],subject['study_time'],subject['study_performance'],subject['composite_scores']],
#             labels=['加权平均成绩','课题研究时长','课题表现','综合得分'],
#             patch_artist=True,
#             boxprops={'color': 'k'},
#             widths=[0.6,0.6,0.6,0.6],
#             notch=True,
#             whiskerprops={'linestyle':'--'},
#             medianprops={'color':'r'},
#             flierprops = {'markerfacecolor': 'red', 'color': 'black'},
# )
# plt.grid(linestyle='-.')
# plt.show()
# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# sns.set_context('talk', font_scale=1.2)
# # 重新组合dataframe并绘制散布分布
# df = pd.concat([art['composite_scores'],sport['composite_scores'],
#            political['composite_scores'],subject['composite_scores'],
#                 diligence['composite_scores'],consumption['composite_scores']]
#           ,axis=1)
# df.columns=['艺术水平','体育水平','品德水平','知识水平','勤劳水平','消费水平']#修改列名称
# df_normalize = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# g = sns.pairplot(df_normalize,kind='reg',diag_kind='kde')
# plt.savefig("matrix.png", dpi=300,format="png")
# plt.show()
# 观察到各变量都较好地符合正态分布 且变量之间的关系分布较为一致


# X: 两维特征
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    # 核心点
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_]=True
    # 异常点
    anomalies_mask = dbscan.labels_ == -1
    # 边界点
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=350, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask], label=np.unique(dbscan.labels_[core_mask]))
    plt.scatter(anomalies[:, 0], anomalies[:, 1],   c="r", marker="x", s=40, label="异常点")
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".",label="边界点")

    if show_xlabels:
        plt.xlabel("$X_1$", fontsize=12)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$X_2$", fontsize=12, rotation=90)
    else:
        plt.tick_params(labelleft=False)
    # plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)
    plt.legend(loc='best', edgecolor='k')

# # 对每个df进行DBSCAN聚类 特征根据指标权重进行二次计算
# 知识水平
dbscan = DBSCAN(eps=0.3, min_samples=7, metric='wminkowski', p=2, metric_params={"w": [0.4,0.1,0.1,0.1,0.1,0.1,0.1]})
subject = subject.drop(['Unnamed: 0','student_id'],axis=1).fillna(0)
# 分析知识水平的异常点
# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# plt.boxplot(x=[subject['subject_ave'],subject['study_time'],subject['study_performance'],subject['composite_scores']],
#             labels=['加权平均成绩','课题研究时长','课题表现','综合得分'],
#             patch_artist=True,
#             boxprops={'color': 'k'},
#             widths=[0.6,0.6,0.6,0.6],
#             notch=True,
#             whiskerprops={'linestyle':'--'},
#             medianprops={'color':'r'},
#             flierprops = {'markerfacecolor': 'red', 'color': 'black'},
# )
# plt.grid(linestyle='-.')
# plt.savefig("subject_box.png", dpi=300,format="png")
# plt.show()

subject_sd = StandardScaler().fit_transform(subject)
subject_sd = pd.DataFrame(data=subject_sd,columns=list(subject.columns))
dbscan.fit(subject_sd)
label = dbscan.labels_
# from sklearn.decomposition import IncrementalPCA
# subject_pca = IncrementalPCA(2).fit_transform(subject)
# plot_dbscan(dbscan,subject_pca,size=100)
# plt.savefig("subject.png", dpi=300,format="png")
# plt.show()

### 建立符号标签体系  #########
dict_x ={
    -1:'A2',
    0:'A1',
    1:'A2',
}
labels = list(dbscan.labels_)
labels_subject = [dict_x[x] for x in labels]
print(len(labels_subject))

### 统计每个类簇的人数以及得分矩均值
# score_mean0=subject['composite_scores'][labels.index(0)].mean()
# score_mean1=subject['composite_scores'][labels.index(1)].mean()
# peple_rate0 =labels.count(0)/len(labels)
# peple_rate1 =labels.count(1)/len(labels)

## 统计轮廓系数 ###########
# sub_sil=[]
# for i in range(sil_num):
#     dbscan = DBSCAN(eps=0.1, min_samples=i, metric='wminkowski', p=2, metric_params={"w": [0.4,0.1,0.1,0.1,0.1,0.1,0.1]})
#     dbscan.fit(subject)
#     sub_sil.append(metrics.silhouette_score(subject_sd,dbscan.labels_,metric='euclidean'))


########  艺术类 ############
dbscan = DBSCAN(eps=0.3, min_samples=4, metric='wminkowski', p=2, metric_params={"w": [0.5,0.3,0.2]})
art_1 = art.drop(['Unnamed: 0','student_id','composite_scores'],axis=1)
from sklearn.preprocessing import  StandardScaler
art_sd = StandardScaler().fit_transform(art_1)
art_sd = pd.DataFrame(data=art_sd,columns=list(art_1.columns))
art_pca = IncrementalPCA(2).fit_transform(art_sd)
dbscan.fit(art_sd)
# plot_dbscan(dbscan,art_pca,size=100)
# plt.savefig("art.png", dpi=300,format="png")
# plt.show()

### 建立符号标签体系  #########
dict_x ={
    -1:'B4',
    0:'B1',
    1:'B2',
    2:'B3',
    3:'B4'
}
labels = list(dbscan.labels_)
labels_art = [dict_x[x] for x in labels]
print(len(labels_art))

## 统计每个类簇的人数以及得分矩均值
# labels = list(dbscan.labels_)
# score_mean0=art['composite_scores'][labels.index(0)].mean()
# score_mean1=art['composite_scores'][labels.index(1)].mean()
# score_mean2=art['composite_scores'][labels.index(2)].mean()
# score_mean3=art['composite_scores'][labels.index(3)].mean()
#
# peple_rate0 =labels.count(0)/len(labels)
# peple_rate1 =labels.count(1)/len(labels)
# peple_rate3 =labels.count(2)/len(labels)
# peple_rate4 =labels.count(3)/len(labels)

# 控制minPts
sil_num=20
#控制半径
sil_radius = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]

### 统计轮廓系数 ###########
# art_sil=[]
# for i in sil_radius:
#     dbscan = DBSCAN(eps=i, min_samples=5, metric='wminkowski', p=2, metric_params={"w": [0.5,0.3,0.2]})
#     dbscan.fit(art)
#     art_sil.append(metrics.silhouette_score(art,dbscan.labels_,metric='euclidean'))

####### 体育类 ##############################
dbscan = DBSCAN(eps=0.22, min_samples=5, metric='wminkowski', p=2, metric_params={"w": [0.4,0.1,0.1,0.2,0.2]})
sport = sport.drop(['Unnamed: 0','student_id','composite_scores'],axis=1).fillna(0)
# #
sport_sd = StandardScaler().fit_transform(sport)
sport_sd = pd.DataFrame(data=sport_sd,columns=list(sport.columns))
sport_pca = IncrementalPCA(2).fit_transform(sport_sd)
dbscan.fit(sport_sd)
# plot_dbscan(dbscan,sport_pca,size=100)
# plt.savefig("political.png", dpi=300,format="png")
# plt.show()

### 建立符号标签体系  #########
dict_x ={
    -1:'C3',
    0:'C1',
    1:'C2',
    2:'C3'
}
labels = list(dbscan.labels_)
labels_sport = [dict_x[x] for x in labels]
print(len(labels_sport))


# labels = list(dbscan.labels_)
# score_mean0=art['composite_scores'][labels.index(0)].mean()
# score_mean1=art['composite_scores'][labels.index(1)].mean()
# score_mean2=art['composite_scores'][labels.index(2)].mean()
# peple_rate0 =labels.count(0)/len(labels)
# peple_rate1 =labels.count(1)/len(labels)
# peple_rate3 =labels.count(2)/len(labels)



### 统计轮廓系数 ###########
# sport_sil=[]
# for i in sil_radius:
#     dbscan = DBSCAN(eps=i, min_samples=5, metric='wminkowski', p=2, metric_params={"w": [0.4,0.1,0.1,0.2,0.2]})
#     dbscan.fit(sport)
#     sport_sil.append(metrics.silhouette_score(sport,dbscan.labels_,metric='euclidean'))

####### 品德类 ##############################
# dbscan = DBSCAN(eps=0.15, min_samples=3, metric='wminkowski', p=2, metric_params={"w": [0.4,0.1,0.1,0.2,0.2]})
# political = political.drop(['Unnamed: 0','student_id','composite_scores'],axis=1).fillna(0)
# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# plt.boxplot(x=[political['score'],political['number'],art['ave_score'],political['composite_scores']],
#             labels=['学科得分','活动次数','活动得分','综合得分'],
#             patch_artist=True,
#             boxprops={'color': 'k'},
#             widths=[0.6,0.6,0.6,0.6],
#             notch=True,
#             whiskerprops={'linestyle':'--'},
#             medianprops={'color':'r'},
#             flierprops = {'markerfacecolor': 'red', 'color': 'black'},
# )
# plt.grid(linestyle='-.')
# plt.savefig("political_box.png", dpi=300,format="png")
# plt.show()
# political_sd = StandardScaler().fit_transform(political)
# political_sd = pd.DataFrame(data=political_sd,columns=list(political.columns))
# political_pca = IncrementalPCA(2).fit_transform(political_sd)
# dbscan.fit(political_sd)
# plot_dbscan(dbscan,political_pca,size=100)
# plt.savefig("political.png", dpi=300,format="png")
# plt.show()
#
# labels = list(dbscan.labels_)
# score_mean0=art['composite_scores'][labels.index(0)].mean()
# score_mean1=art['composite_scores'][labels.index(1)].mean()
#
# peple_rate0 =labels.count(0)/len(labels)
# peple_rate1 =labels.count(1)/len(labels)


### 建立符号标签体系  #########
dict_x ={
    -1:'D3',
    0:'D1',
    1:'D2',
    2:'D3'
}
labels = list(dbscan.labels_)
labels_political = [dict_x[x] for x in labels]
print(len(labels_political))

### 统计轮廓系数 ###########
# pol_sil=[]
# for i in sil_radius:
#     dbscan = DBSCAN(eps=i, min_samples=5, metric='wminkowski', p=2, metric_params={"w": [0.4,0.1,0.1,0.2,0.2]})
#     dbscan.fit(political)
#     pol_sil.append(metrics.silhouette_score(political,dbscan.labels_,metric='euclidean'))
#


####### 勤劳类 ##############################
dbscan = DBSCAN(eps=2.1, min_samples=4, metric='wminkowski', p=2, metric_params={"w": [0.4,0.4,0.1,0.1]})
diligence = pd.read_csv('diligence.csv').drop(['Unnamed: 0','student_id','composite_scores'],axis=1).fillna(0)
diligence_sd = StandardScaler().fit_transform(diligence)
diligence_sd = pd.DataFrame(data=diligence_sd,columns=list(diligence.columns))
diligence_pca = PCA(2).fit_transform(diligence_sd)
dbscan.fit(diligence)
# plot_dbscan(dbscan,diligence_pca,size=100)
# plt.savefig("diligence.png", dpi=300,format="png")
# plt.show()
#
#
# labels = list(dbscan.labels_)
# score_mean0=art['composite_scores'][labels.index(0)].mean()
# score_mean1=art['composite_scores'][labels.index(1)].mean()
# score_mean2=art['composite_scores'][labels.index(2)].mean()
# score_mean3=art['composite_scores'][labels.index(3)].mean()
#
# peple_rate0 =labels.count(0)/len(labels)
# peple_rate1 =labels.count(1)/len(labels)
# peple_rate3 =labels.count(2)/len(labels)
# peple_rate4 =labels.count(3)/len(labels)

### 建立符号标签体系  #########
dict_x ={
    -1:'E4',
    0:'E1',
    1:'E2',
    2:'E3',
    3:'E4'
}
labels = list(dbscan.labels_)
labels_diligence = [dict_x[x] for x in labels]
print(len(labels_diligence))


### 计算轮廓系数 #####
# dili_sil=[]
# dili_sil1=[]
# dili_sil2=[]
# dili_sil3=[]
# for i in sil_radius:
    # dbscan = DBSCAN(eps=0.3, min_samples=i, metric='wminkowski', p=2, metric_params={"w": [0.4,0.4,0.1,0.1]})
    # dbscan1 = DBSCAN(eps=0.7, min_samples=i, metric='wminkowski', p=2, metric_params={"w": [0.4,0.4,0.1,0.1]})
    # dbscan2 = DBSCAN(eps=1.2, min_samples=i, metric='wminkowski', p=2, metric_params={"w": [0.4,0.4,0.1,0.1]})
    # dbscan3 = DBSCAN(eps=i, min_samples=3, metric='wminkowski', p=2, metric_params={"w": [0.4,0.4,0.1,0.1]})
    # dbscan.fit(diligence)
    # dbscan1.fit(diligence)
    # dbscan2.fit(diligence)
    # dbscan3.fit(diligence)
    # dili_sil.append(metrics.silhouette_score(diligence,dbscan.labels_,metric='euclidean'))
    # dili_sil1.append(metrics.silhouette_score(diligence,dbscan.labels_,metric='euclidean'))
    # dili_sil2.append(metrics.silhouette_score(diligence,dbscan.labels_,metric='euclidean'))
    # dili_sil3.append(metrics.silhouette_score(diligence,dbscan3.labels_,metric='euclidean'))


plt.figure(1)
plt.xlabel(r'$\delta$')
plt.ylabel('Silhouette Coefficient')
# plt.plot(range(4),dili_sil,marker='o',markersize=5)
# plt.plot(range(4),dili_sil1,marker='o',markersize=5)
# plt.plot(range(4),dili_sil2,marker='o',markersize=5)
# plt.plot(dili_sil3,marker='o',markersize=5)


####### 消费类 ##############################
dbscan = DBSCAN(eps=12.25, min_samples=5, metric='wminkowski', p=2, metric_params={"w": [0.7,0.2,0.1]})
# consumption = pd.read_csv('consumption.csv')
# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# plt.boxplot(x=[consumption['term_number'],consumption['month_number'],consumption['month_amount'],consumption['composite_scores']],
#             labels=['学期消费次数','月消费次数','月消费金额','综合得分'],
#             patch_artist=True,
#             boxprops={'color': 'k'},
#             widths=[0.6,0.6,0.6,0.6],
#             notch=True,
#             whiskerprops={'linestyle':'--'},
#             medianprops={'color':'r'},
#             flierprops = {'markerfacecolor': 'red', 'color': 'black'},
# )
# plt.grid(linestyle='-.')
# plt.savefig("consumption_box.png", dpi=300,format="png")
# plt.show()
consumption = pd.read_csv('consumption.csv').drop(['Unnamed: 0','student_id','composite_scores'],axis=1).fillna(0)
consumption_sd = StandardScaler().fit_transform(consumption)
consumption_sd = pd.DataFrame(data=consumption_sd,columns=list(consumption.columns))
consumption_pca = PCA(2).fit_transform(consumption_sd)
dbscan.fit(consumption)
# plot_dbscan(dbscan,consumption_pca,size=100)
# plt.savefig("consumption.png", dpi=300,format="png")
# plt.show()
### 建立符号标签体系  #########
dict_x ={
    -1:'F7',
    0:'F1',
    1:'F2',
    2:'F3',
    3:'F4',
    4:'F5',
    5:'F6',
    6:'F7'
}
labels = list(dbscan.labels_)
labels_consumption = [dict_x[x] for x in labels]
print(len(labels_consumption))

# labels = list(dbscan.labels_)
# score_mean0=art['composite_scores'][labels.index(0)].mean()
# score_mean1=art['composite_scores'][labels.index(1)].mean()
# score_mean2=art['composite_scores'][labels.index(2)].mean()
# score_mean3=art['composite_scores'][labels.index(3)].mean()
# score_mean4=art['composite_scores'][labels.index(4)].mean()
# score_mean5=art['composite_scores'][labels.index(5)].mean()
# score_mean6=art['composite_scores'][labels.index(6)].mean()
#
# peple_rate0 =labels.count(0)/len(labels)
# peple_rate1 =labels.count(1)/len(labels)
# peple_rate2 =labels.count(2)/len(labels)
# peple_rate3 =labels.count(3)/len(labels)
# peple_rate4 =labels.count(4)/len(labels)
# peple_rate5 =labels.count(5)/len(labels)
# peple_rate6 =labels.count(6)/len(labels)


### 统计轮廓系数 ###########
# con_sil=[]
# for i in range(12):
#     dbscan = DBSCAN(eps=0.01, min_samples=i, metric='wminkowski', p=2, metric_params={"w": [0.7,0.2,0.1]})
#     dbscan.fit(consumption_sd)
#     con_sil.append(metrics.silhouette_score(consumption_sd,dbscan.labels_,metric='euclidean'))


# all_sil = [art_sil,sport_sil,pol_sil,dili_sil]
# plt.figure(figsize=(10, 5))  # 设置画布的尺寸
# plt.plot(sub_sil,'-o',linewidth=2)
# plt.plot(dili_sil,'-o',linewidth=2)
# plt.plot(art_sil,'-o',linewidth=2)
# plt.plot(sport_sil,'-o',linewidth=2)
# plt.plot(pol_sil,'-o',linewidth=2)
# # plt.plot(con_sil,'-o',linewidth=2)
# plt.xlabel(r'$\delta$')
# plt.ylabel('Silhouette Coefficient')
# plt.legend([r'勤劳程度',r'艺术水平',r'体育水平',r'品德水平'])
# plt.grid(linestyle='-.')
# plt.savefig("Silhouette.png", dpi=300,format="png")
# plt.show()

#####################################

### 根据符号list 重新生成 dataframe

student_id = [i for i in range(14452,14452+342)]
dict_new = {
    "ID":student_id,
    '知识':labels_subject[:342],
    '艺术':labels_art[:342],
    "体育":labels_sport[:342],
    "品德":labels_political[:342],
    "劳动":labels_diligence[:342],
    "消费":labels_consumption[:342]
}
dataframe_sys = pd.DataFrame(dict_new)
dataframe_sys.to_csv('sysboml.csv')


