import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets._samples_generator import make_blobs
#### 数据加载+二次加工+计算不同维度指标+箱装图分析异常点+散布矩阵分析+归一化+聚类+FP-Tree树关联挖掘

########################################################### 品德  #######################################################

import pandas as pd
action_info =  pd.read_csv('../data_generate/action.csv')
student_info = pd.read_csv('../data_generate/student_info.csv')
subject_info = pd.read_csv('../data_generate/knowledge.csv')
# # 考核成绩指标
political_info = subject_info.loc[subject_info['subject']=='政治'][['student_id','program_grade']].set_index(['student_id']).sort_index()
print(len(political_info))
# #活动平均得分
# political_action_info = action_info.loc[(action_info['action_type']=='思政比赛') | (action_info['action_type'] =='志愿服务')].reset_index(drop=True)
# political_action_info['action_score'] = political_action_info.apply(lambda x: x['action_level'] * x['action_performance'], axis=1)
# political_ave_score = political_action_info.groupby(['student_id'])['action_score'].mean()
# print(len(political_ave_score))
# #  活动参与频次
# political_number = political_action_info.groupby(['student_id'])['action_type'].count()
# print(len(political_number))
# # 拼接生成二级评价表
# # 长度不一致 说明部分学生没有参与品德相关的活动 在拼接时活动得分和参与频次置0
# political_second_tab = {
#     "student_id":[],
#     "ave_score":[],
#     "number":[],
#     "grade":[]
# }
# for index, row in political_info.iterrows():
#     political_second_tab["student_id"].append(index)
#     political_second_tab["grade"].append(row['program_grade'])
#     if index in political_ave_score.index:
#         political_second_tab["ave_score"].append(political_ave_score[index])
#     else:
#         political_second_tab["ave_score"].append(0)
#     if index in political_number.index:
#         political_second_tab["number"].append(political_number[index])
#     else:
#         political_second_tab["number"].append(0)
# political_second = pd.DataFrame(political_second_tab)
# # 根据指标权重计算
# political_second['composite_scores']=political_second.apply(lambda x: x['ave_score']*0.7+x['number']*0.2+x['grade']*0.1,axis=1)
# political_second.to_csv('political.csv')

########################################################## 艺术  #######################################################
# # 考核成绩指标
# art_info = subject_info.loc[(subject_info['subject']=='音乐')|(subject_info['subject']=='美术')][['student_id','program_grade']].set_index(['student_id']).sort_index()
# art_info = art_info.groupby(['student_id'])['program_grade'].mean()
# print(len(art_info))
# # 活动平均得分
# art_action_info = action_info.loc[(action_info['action_type']=='艺术活动')].reset_index(drop=True)
# art_action_info['action_score'] = art_action_info.apply(lambda x: x['action_level'] * x['action_performance'], axis=1)
# art_ave_score = art_action_info.groupby(['student_id'])['action_score'].mean()
# print(len(art_ave_score))
# # # 活动参与频次
# art_number = art_action_info.groupby(['student_id'])['action_type'].count()
# print(len(art_number))
# # 拼接生成二级评价表
# art_second_tab = {
#     "student_id":[],
#     "ave_score":[],
#     "number":[],
#     "grade":[]
# }
# for index, value in art_info.items():
#     art_second_tab["student_id"].append(index)
#     art_second_tab["grade"].append(value)
#     if index in art_ave_score.index:
#         art_second_tab["ave_score"].append(art_ave_score[index])
#     else:
#         art_second_tab["ave_score"].append(0)
#     if index in art_number.index:
#         art_second_tab["number"].append(art_number[index])
#     else:
#         art_second_tab["number"].append(0)
# art_second = pd.DataFrame(art_second_tab)
# # 根据指标权重计算
# art_second['composite_scores']=art_second.apply(lambda x: x['ave_score']*0.7+x['number']*0.2+x['grade']*0.1,axis=1)
# art_second.to_csv('art.csv')
########################################################### 体育  #######################################################
# 考核成绩指标
# exam = pd.read_csv('5_chengji.csv')
# sport_info = exam.loc[exam['mes_sub_name']=='政治']
# sport_info['mes_Z_Score'].fillna(sport_info['mes_Z_Score'].min())
# sport_info.fillna(0)
# mes_z = sport_info.groupby(['mes_StudentID'])['mes_Z_Score'].mean()
# mes_T = sport_info.groupby(['mes_StudentID'])['mes_T_Score'].mean()
# mes_dengdi = sport_info.groupby(['mes_StudentID'])['mes_dengdi'].mean()
# mes_dengdi[mes_dengdi<0] = 0
# mes_score = sport_info.groupby(['mes_StudentID'])['mes_Score'].mean()
# political_action_info = action_info.loc[(action_info['action_type']=='思政比赛') | (action_info['action_type'] =='志愿服务')].reset_index(drop=True)
# # sport_action_info = action_info.loc[(action_info['action_type']=='体育比赛')].reset_index(drop=True)
# # 活动参与频次
# political_number = political_action_info.groupby(['student_id'])['action_type'].count()
# print(len(political_number))
#
# # 拼接生成二级评价表
# sport_second_tab = {
#     "student_id":[],
#     "score":[],
#     "number":[],
#     "mes_z":[],
#     "mes_T":[],
#     "mes_dengdi":[]
# }
# for index, value in political_number.items():
#     sport_second_tab["student_id"].append(index)
#     sport_second_tab["number"].append(value)
#     if index in mes_z.index:
#         sport_second_tab["score"].append(mes_score[index])
#         sport_second_tab["mes_z"].append(mes_z[index])
#         sport_second_tab["mes_T"].append(mes_T[index])
#         sport_second_tab["mes_dengdi"].append(mes_dengdi[index])
#     else:
#         sport_second_tab["score"].append(0)
#         sport_second_tab["mes_z"].append(0)
#         sport_second_tab["mes_T"].append(0)
#         sport_second_tab["mes_dengdi"].append(0)
#
# sport_second = pd.DataFrame(sport_second_tab)
# # 根据指标权重计算
# sport_second['composite_scores']=sport_second.apply(lambda x: x['score']*0.7+x['number']*0.3,axis=1)
# sport_second.to_csv('political.csv')
########################################################### 智力和知识水平  #######################################################
# 加权平均成绩
# subject_list=['语文','数学','英语','物理','化学','生物','政治','历史','地理']
# credits_list=[3, 3, 3, 2, 2, 2, 1, 1, 1]
# exam = pd.read_csv('5_chengji.csv')
# exam['mes_Z_Score'].fillna(exam['mes_Z_Score'].min())
# exam['mes_T_Score'].fillna(0)
#
# exam.drop(exam[~(exam['mes_sub_name'].isin(subject_list))].index, inplace=True)
# exam['credit'] = exam.apply(lambda x:credits_list[subject_list.index(x['mes_sub_name'])],axis=1)
# exam["score_1"] = exam.apply(lambda x:x['credit']*x['mes_Score'],axis=1)
# subject_ave = exam.groupby('mes_StudentID')['score_1'].sum()/exam.groupby('mes_StudentID')['credit'].sum()
# mes_dengdi = exam.groupby(['mes_StudentID'])['mes_dengdi'].mean()
# mes_z = exam.groupby(['mes_StudentID'])['mes_Z_Score'].mean()
# mes_T = exam.groupby(['mes_StudentID'])['mes_T_Score'].mean()
# mes_dengdi[mes_dengdi<0] = 0
# # 课题研究时长
# study_time = subject_info.groupby('student_id')['study_time'].mean()
# # 课题评分 随机生成
# subject_second_tab = {
#     "student_id":[],
#     "subject_ave":[],
#     "study_time":[],
#     "mes_dengdi":[],
#     "mes_z":[],
#     "mes_T":[],
#     "study_performance":[]
# }
# for index, value in study_time.items():
#     subject_second_tab["student_id"].append(index)
#     subject_second_tab["study_time"].append(study_time[index])
#     if index in subject_ave.index:
#         subject_second_tab["mes_dengdi"].append(mes_dengdi[index])
#         subject_second_tab["subject_ave"].append(subject_ave[index])
#         subject_second_tab["mes_z"].append(mes_z[index])
#         subject_second_tab["mes_T"].append(mes_T[index])
#     else:
#         subject_second_tab["mes_dengdi"].append(0)
#         subject_second_tab["subject_ave"].append(0)
#         subject_second_tab["mes_z"].append(0)
#         subject_second_tab["mes_T"].append(0)
#     subject_second_tab["study_performance"].append(np.random.randint(50,100))
# # 根据指标权重计算
# subject_second = pd.DataFrame(subject_second_tab)
# subject_second['composite_scores'] = subject_second.apply(lambda x: x['subject_ave']*0.7+x['study_time']*0.1+x['study_performance']*0.2,axis=1)
# subject_second.to_csv('subject.csv')

########################################################### 消费水平 #######################################################
# consumption_info=pd.read_csv('../data_generate/consumption.csv')
# consumption_info['composite_scores'] = consumption_info.apply(lambda x: x['term_amount']*0.4+x['month_number']*0.2+x['month_amount']*0.3,axis=1)
# consumption_info.to_csv('consumption.csv')
########################################################### 勤劳水平 #######################################################


diligence = pd.read_csv('../data_generate/diligence.csv').set_index(['student_id']).sort_index()

kaoqin = pd.read_csv('3_kaoqin.csv')
# "controler_id","controler_name","control_task_order_id","control_task_name"
# "001001","迟到_晚到","100000","默认信息"
# "001001","迟到_晚到","100100","早上迟到"
# "001001","迟到_晚到","100200","晚到学校"
# "001001","迟到_晚到","100300","晚自修迟到"
# "001002","校徽_早退","200000","默认信息"
# "001002","校徽_早退","200100","校徽校服"
# "001002","校徽_早退","200200","请假离校"
# "001003","操场考勤机","300000","默认信息"
# "001003","操场考勤机","300100","住宿早晨锻炼"
# "001003","操场考勤机","300200","课间操请假"
# "0099001","迟到[移动考勤机]","9900100","默认信息"
# "0099002","校服[移动考勤机]","9900200","默认信息"
# "0099003","早退[移动考勤机]","9900300","默认信息"
# "0099004","离校[移动考勤机]","9900400","离校考勤"
# "0099005","进校[移动考勤机]","9900500","进校考勤"

id_list= [100100,100200,100300,200200,300200]
late_num = kaoqin.drop(kaoqin[kaoqin['control_task_order_id'].isin(id_list)].index).groupby(['bf_studentID'])['control_task_order_id'].count()


early_num = kaoqin.drop(kaoqin[~(kaoqin['control_task_order_id']
                             .isin([300100]))].index)\
                             .groupby(['bf_studentID'])['control_task_order_id'].count()
late_num=late_num.reset_index(drop=True)
early_num=early_num.reset_index(drop=True)

diligence_second_tab = {
    "student_id":[],
    "normal_at_num":[],
    "early_at_num":[],
    "practice_num":[],
    "score_practice":[],
}
for i in range(min(len(late_num),len(early_num))):
    diligence_second_tab["student_id"].append(i)
    diligence_second_tab["practice_num"].append(np.random.randint(1,10))
    diligence_second_tab["score_practice"].append(np.random.randint(10,100))
    diligence_second_tab["normal_at_num"].append(late_num[i])
    diligence_second_tab["early_at_num"].append(early_num[i])



# for i, v in late_num:
#     diligence_second_tab["student_id"].append(i)
#     diligence_second_tab["practice_num"].append(np.random.randint(1,10))
#     diligence_second_tab["score_practice"].append(np.random.randint(10,100))
#     diligence_second_tab["normal_at_num"].append(late_num[index])
#     diligence_second_tab["early_at_num"].append(early_num[index])
#
# for index, row in diligence.iterrows():
#     diligence_second_tab["student_id"].append(index)
#     diligence_second_tab["practice_num"].append(np.random.randint(1,10))
#     diligence_second_tab["score_practice"].append(np.random.randint(10,100))
#     diligence_second_tab["normal_at_num"].append(late_num[index])
#     diligence_second_tab["early_at_num"].append(early_num[index])
#     else:
#         diligence_second_tab["normal_at_num"].append(0)
#         diligence_second_tab["early_at_num"].append(0)

diligence_second = pd.DataFrame(diligence_second_tab)
diligence_second['composite_scores'] = diligence_second.apply(lambda x: x['normal_at_num']*0.2+x['early_at_num']*0.3+
                                                          x['practice_num']*0.2+x['score_practice']*0.3,axis=1)
diligence_second.to_csv('diligence.csv')







