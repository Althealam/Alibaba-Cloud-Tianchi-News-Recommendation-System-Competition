import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 用户历史行为相关特征
def create_feature(users_id, recall_list, click_hist_df,  articles_info, articles_emb, user_emb=None, N=1):
    """
    基于用户的历史行为做相关特征
    :param users_id: 用户id
    :param recall_list: 对于每个用户召回的候选文章列表
    :param click_hist_df: 用户的历史点击信息
    :param articles_info: 文章信息
    :param articles_emb: 文章的embedding向量, 这个可以用item_content_emb, item_w2v_emb, item_youtube_emb
    :param user_emb: 用户的embedding向量， 这个是user_youtube_emb, 如果没有也可以不用， 但要注意如果要用的话， articles_emb就要用item_youtube_emb的形式， 这样维度才一样
    :param N: 最近的N次点击  由于testA日志里面很多用户只存在一次历史点击， 所以为了不产生空值，默认是1
    """
    
    # 建立一个二维列表保存结果， 后面要转成DataFrame
    all_user_feas = []
    i = 0
    for user_id in tqdm(users_id):
        # 该用户的最后N次点击
        hist_user_items = click_hist_df[click_hist_df['user_id']==user_id]['click_article_id'][-N:]
        
        # 遍历该用户的召回列表
        for rank, (article_id, score, label) in enumerate(recall_list[user_id]):
            # 该文章建立时间, 字数
            a_create_time = articles_info[articles_info['article_id']==article_id]['created_at_ts'].values[0]
            a_words_count = articles_info[articles_info['article_id']==article_id]['words_count'].values[0]
            single_user_fea = [user_id, article_id]
            # 计算与最后点击的商品的相似度的和， 最大值和最小值， 均值
            sim_fea = []
            time_fea = []
            word_fea = []
            # 遍历用户的最后N次点击文章
            for hist_item in hist_user_items:
                b_create_time = articles_info[articles_info['article_id']==hist_item]['created_at_ts'].values[0]
                b_words_count = articles_info[articles_info['article_id']==hist_item]['words_count'].values[0]
                
                sim_fea.append(np.dot(articles_emb[hist_item], articles_emb[article_id]))
                time_fea.append(abs(a_create_time-b_create_time))
                word_fea.append(abs(a_words_count-b_words_count))
                
            single_user_fea.extend(sim_fea)      # 相似性特征
            single_user_fea.extend(time_fea)    # 时间差特征
            single_user_fea.extend(word_fea)    # 字数差特征
            single_user_fea.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea) / len(sim_fea)])  # 相似性的统计特征
            
            if user_emb:  # 如果用户向量有的话， 这里计算该召回文章与用户的相似性特征 
                single_user_fea.append(np.dot(user_emb[user_id], articles_emb[article_id]))
                
            single_user_fea.extend([score, rank, label])    
            # 加入到总的表中
            all_user_feas.append(single_user_fea)
    
    # 定义列名
    id_cols = ['user_id', 'click_article_id']
    sim_cols = ['sim' + str(i) for i in range(N)]
    time_cols = ['time_diff' + str(i) for i in range(N)]
    word_cols = ['word_diff' + str(i) for i in range(N)]
    sat_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean']
    user_item_sim_cols = ['user_item_sim'] if user_emb else []
    user_score_rank_label = ['score', 'rank', 'label']
    cols = id_cols + sim_cols + time_cols + word_cols + sat_cols + user_item_sim_cols + user_score_rank_label
            
    # 转成DataFrame
    df = pd.DataFrame( all_user_feas, columns=cols)
    
    return df


# 区分用户活跃度的特征
def active_level(all_data,cols):
    """
        制作区分用户活跃度的特征
        :param all_data: 数据集
        :cols: 用到的特征列
    """
    data=all_data[cols]
    data.sort_values(['user_id','click_timestamp'],inplace=True)
    user_act=pd.DataFrame(data.groupby('user_id',as_index=False)[['click_article_id','click_timestamp']].\
                          agg({'click_article_id':np.size,'click_timestamp':{list}}).values,columns=['user_id','click_size','click_timestamp'])
    
    # 计算时间间隔的均值
    def time_diff_mean(l):
        if len(l)==1:
            return 1
        else:
            return np.mean([j-i for i,j in list(zip(l[:-1],l[1:]))])
    
    user_act['time_diff_mean']=user_act['click_timestamp'].apply(lambda x: time_diff_mean(x))
    
    # 点击次数取倒数
    user_act['click_size']=1/user_act['click_size']
    
    # 两者归一化
    user_act['click_size']=(user_act['click_size']-user_act['click_size'].min())/(user_act['click_size'].max()-user_act['click_size'].min())
    user_act['time_diff_mean']=(user_act['time_diff_mean']-user_act['time_diff_mean'].min())/(user_act['time_diff_mean'].max()-user_act['time_diff_mean'].min())
    user_act['active_level']=user_act['click_size']+user_act['time_diff_mean']
    
    user_act['user_id']=user_act['user_id'].astype('int')
    del user_act['click_timestamp']
    
    return user_act


# 衡量文章热度的特征
def hot_level(all_data,cols):
    """
        制作衡量文章热度的特征
        :param all_data: 数据集
        :param cols: 用到的特征列
    """
    data=all_data[cols]
    data.sort_values(['click_article_id','click_timestamp'],inplace=True)
    article_hot=pd.DataFrame(data.groupby('click_article_id',as_index=False)[['user_id','click_timestamp']].\
                             agg({'user_id':np.size,'click_timestamp':{list}}).values,columns=['click_article_id','user_num','click_timestamp'])
    
    # 计算被点击时间间隔的均值
    def time_diff_mean(l):
        if len(l)==1:
            return 1
        else:
            return np.mean([j-i for i,j in list(zip(l[:-1],l[1:]))])
        
    article_hot['time_diff_mean']=article_hot['click_timestamp'].apply(lambda x: time_diff_mean(x))
    
    # 点击次数取倒数
    article_hot['user_num']=1/article_hot['user_num']
    
    # 两者归一化
    article_hot['user_num']=(article_hot['user_num']-article_hot['user_num'].min())/(article_hot['user_num'].max()-article_hot['user_num'].min())
    article_hot['time_diff_mean']=(article_hot['time_diff_mean']-article_hot['time_diff_mean'].min())/(article_hot['time_diff_mean'].max()-article_hot['time_diff_mean'].min())
    article_hot['hot_level']=article_hot['user_num']+article_hot['time_diff_mean']
    
    article_hot['click_article_id']=article_hot['click_article_id'].astype('int')
    
    del article_hot['click_timestamp']
    return article_hot

# 用户的设备习惯
def device_fea(all_data,cols):
    """
        制作用户的设备特征
        :param all_data: 数据集
        :param cols: 用到的特征列
    """
    user_device_info=all_data[cols]
    
    # 用众书来表示每个用户的设备信息
    user_device_info=user_device_info.groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()
    
    return user_device_info


# 用户的时间习惯
def user_time_hob_fea(all_data,cols):
    """
        制作用户的时间习惯特征
        :param all_data: 数据集
        :param cols: 用到的特征列
    """
    user_time_hob_info=all_data[cols]

    # 将时间戳进行归一化
    mm=MinMaxScaler()
    user_time_hob_info['click_timestamp']=mm.fit_transform(user_time_hob_info[['click_timestamp']])
    user_time_hob_info['created_at_ts']=mm.fit_transform(user_time_hob_info[['created_at_ts']])

    user_time_hob_info=user_time_hob_info.groupby('user_id').agg('mean').reset_index()

    user_time_hob_info.rename(columns={'click_timestamp':'user_time_hob1','created_at_ts':'user_time_hob2'},inplace=True)
    return user_time_hob_info

# 用户的主题爱好
# 先把用户点击的文章属于的主题转换为一个列表
def user_cat_hob_fea(all_data,cols):
    """
        用户的主题爱好
        :param all_data: 数据集
        :param cols: 用到的特征列
    """
    user_category_hob_info=all_data[cols]
    user_category_hob_info=user_category_hob_info.groupby('user_id').agg({'category_id':list}).reset_index()

    user_cat_hob_info=pd.DataFrame()
    user_cat_hob_info['user_id']=user_category_hob_info['user_id']
    user_cat_hob_info['cate_list']=user_category_hob_info['category_id']

    return user_cat_hob_info    