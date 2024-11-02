# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果
from tqdm import tqdm
import pickle
import datetime
import pandas as pd

def get_click_article_ids_set(all_click_df):
    return set(all_click_df.click_article_id.values)

def cold_start_items(user_recall_items_dict,user_hist_item_typs_dict,user_hist_item_words_dict,\
                    user_last_item_created_time_dict,item_type_dict,item_words_dict,
                    item_created_time_dict,click_article_ids_set,recall_item_num):
    """
        冷启动的情况下召回一些文章
        :param user_recall_items_dict: 基于内容embedding相似性召回来的文章字典，{user1:[(item1,item2),...],...}
        :param user_hist_item_typs_dict: 字典，用户点击的文章的主题映射
        :param user_hist_item_words_dict: 字典，用户点击的历史文章的字数映射
        :param user_last_item_created_time_dict: 字典，用户点击的历史文章创建时间映射
        :param item_type_dict: 字典，文章主题映射
        :param item_words_dict: 字典，文章字数映射
        :param item_created_time_dict: 字典，文章创建时间映射
        :param click_article_ids_set: 集合，用户点击过的文章，也就是日志里面出现过的文章
        :param recall_item_num: 召回文章的数量（指的是没有出现在日志里面的文章数量） 
    """
    cold_start_user_items_dict={}
    for user,item_list in tqdm(user_recall_items_dict.items()):
        cold_start_user_items_dict.setdefault(user,[])
        for item,score in item_list:
            # 获取历史文章信息
            hist_item_type_set=user_hist_item_typs_dict[user]
            hist_mean_words=user_hist_item_words_dict[user]
            hist_last_item_created_time=user_last_item_created_time_dict[user]                
            hist_last_item_created_time=datetime.datetime.fromtimestamp(hist_last_item_created_time)
                
            # 获取当前召回文章的信息
            curr_item_type=item_type_dict[item]
            curr_item_words=item_words_dict[item]
            curr_item_created_time=item_created_time_dict[item]
            curr_item_created_time=datetime.datetime.fromtimestamp(curr_item_created_time)
            if pd.isna(curr_item_created_time):
                print(f"Invalid timestamp for item {item}: {curr_item_created_time}")
                continue  # 或者进行其他处理

                
            # 文章不能出现在用户的历史点击中，然后根据文章主题，文章单词数，文章创建时间进行筛选
            if curr_item_type not in hist_item_type_set or \
                item in click_article_ids_set or \
                abs(curr_item_words-hist_mean_words)>200 or \
                abs((curr_item_created_time-hist_last_item_created_time).days)>90:
                    continue
                    
            cold_start_user_items_dict[user].append((item,score))
            
    # 控制冷启动召回的数量
    cold_start_user_items_dict={k: sorted(v,key=lambda x: x[1],reverse=True)[:recall_item_num]\
                                        for k,v in cold_start_user_items_dict.items()}

    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'    
    pickle.dump(cold_start_user_items_dict,open(save_path+'cold_start_user_items_dict.pkl','wb'))
    return cold_start_user_items_dict