import numpy as np
########################################################### 品德、体育和艺术评价汇总表  #######################################################

#
# 选择一学期内数据生成二级表+分析
import pandas as pd
from collections import Counter
student_info = pd.read_csv('../../data/2_student_info.csv')
print(student_info.columns.to_list())
print(any(student_info['bf_StudentID'].duplicated()))
data_len = len(student_info['bf_StudentID'])

# action_result_dic = {
#     'student_id': [],
#     'action_type':[],
#     'action_number':[],
#     'action_level':[],
#     'action_performance':[]
# }
# for i in range(data_len):
#     number = np.random.randint(1,5)
#     action_type = ['思政比赛', '志愿服务', '体育比赛', '艺术活动']
#     action_type = np.random.choice(action_type, size=number, replace=True)
#     action_count = Counter(action_type)
#     action_number = []
#     for t in action_type:
#         action_number.append(action_count[t])
#     action_result_dic['action_type'].extend(action_type)
#     action_result_dic['action_number'].extend([1]*number)
#     action_result_dic['action_performance'].extend(np.random.choice(range(1,10), size=number, replace=True))
#     action_result_dic['student_id'].extend([student_info['bf_StudentID'][i]]*number)
#     action_result_dic['action_level'].extend(np.random.choice(range(1,5),size=number,replace=True))
# result_dataframe = pd.DataFrame(action_result_dic)
# result_dataframe.to_csv('action.csv')

# ###########################################################  勤劳程度评价汇总表  #######################################################

diligence_result_dic = {
    'student_id': [],
    'normal_at_num':[],
    'early_at_num':[],
    'practice_num':[],
    'score_practice':[],
    'innovation_num':[],
    'score_innovation':[]
}
for i in range(data_len):
    diligence_result_dic['student_id'].append(student_info['bf_StudentID'][i])
    diligence_result_dic['normal_at_num'].append(np.random.randint(20,30))
    diligence_result_dic['early_at_num'].append(np.random.randint(1,10))
    diligence_result_dic['practice_num'].append(np.random.choice(5,size=1,p=[0.3,0.3,0.2,0.1,0.1])[0])
    diligence_result_dic['score_practice'].append(np.random.randint(60,100))
    diligence_result_dic['innovation_num'].append(np.random.choice(5, size=1, p=[0.3, 0.3, 0.2, 0.1, 0.1])[0])
    diligence_result_dic['score_innovation'].append(np.random.randint(60, 100))

result_dataframe = pd.DataFrame(diligence_result_dic)
result_dataframe.to_csv('diligence.csv')
#
#
# ###########################################################  知识水平评价汇总表  #######################################################
# knowledge_result_dic = {
#     'student_id': [],
#     'credits': [],
#     'study_time': [],
#     'program_grade': [],
#     'subject':[]
# }
# subject_list=['语文','数学','英语','物理','化学','生物','政治','历史','地理','音乐','美术','体育']
# credits_list=[3, 3, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0]
# # 每门课都有额外的课题研究时长 和 课题评分
# for i in range(data_len):
#     knowledge_result_dic['subject'].extend(subject_list)
#     knowledge_result_dic['student_id'].extend([student_info['bf_StudentID'][i]] * len(subject_list))
#     knowledge_result_dic['credits'].extend(credits_list)
#     knowledge_result_dic['study_time'].extend(np.random.randint(60,240,size=len(subject_list)))
#     knowledge_result_dic['program_grade'].extend(np.random.randint(50,100,size=len(subject_list)))
# result_dataframe = pd.DataFrame(knowledge_result_dic)
# result_dataframe.to_csv('knowledge.csv')
# ###########################################################  消费水平评价汇总表  #######################################################

consumption_ = pd.read_csv('7_consumption.csv')
def  absolute_value(s):
    return s if(s > 0) else -s
consumption_['MonDeal'] = consumption_['MonDeal'].apply(absolute_value)
deal = consumption_.groupby(['bf_StudentID'])['MonDeal'].sum()
consumption_result_dic = {
    'student_id': [],
    'term_number': [],
    'month_number': [],
    'month_amount': []
}
for i, v in deal.items():
    consumption_result_dic['student_id'].append(i)
    consumption_result_dic['month_number'].append(np.random.randint(30,100))
    consumption_result_dic['term_number'].append(np.random.randint(100,360))
    consumption_result_dic['month_amount'].append(deal[i])
result_dataframe = pd.DataFrame(consumption_result_dic)
result_dataframe['composite_scores'] = result_dataframe.apply(lambda x: x['month_amount']*0.7+x['month_number']*0.2+x['term_number']*0.1,axis=1)
result_dataframe.to_csv('consumption.csv')
