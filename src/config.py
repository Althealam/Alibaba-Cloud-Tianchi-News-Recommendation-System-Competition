import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import YoutubeDNN
from deepmatch.utils import sampledsoftmaxloss,NegativeSampler
warnings.filterwarnings('ignore')
import time
from recall.neg_sample import neg_sample_recall_data

# 获取用户-文章-时间函数
# 根据时间获取用户点击的商品序列: {user1:{item1:time1, item2:time2,...}}
def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('click_timestamp')
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict


# 获取文章-用户-时间函数
# 根据时间获取商品被点击的用户序列 {item1: {user1:time1, user2:time2, ...}}
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))
    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict


# 获取历史和最后一次点击
# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    """
        返回两个df：一个是用户的历史点击记录（不包括最后一次点击）
                  一个是用户的最后一次点击记录
    """
    all_click=all_click.sort_values(by=['user_id','click_timestamp'])
    click_last_df=all_click.groupby('user_id').tail(1)
    
    def hist_func(user_df):
        """
            获取每个用户的历史点击记录（不包括最后一次点击）
            如果用户只有一次点击记录，则直接返回这个记录
            如果用户有多次点击记录，返回除了最后一次点击之外的所有记录
        """
        if len(user_df)==1:
            return user_df
        else:
            return user_df[:-1]
    
    # 获取历史点击
    click_hist_df=all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    
    # click_hist_df：每个用户的历史点击记录
    # click_last_df：包含每个用户的最后一次点击记录
    return click_hist_df, click_last_df

# 获取文章属性
# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段使用
def get_item_info_dict(item_info_df):
    """
        从一个包含文章（或者商品）信息的df中提取每篇文章的特定属性，并将其保存为字典形式
        提取每篇文章的类别id、词数和创建时间，并且将他们保存在字典中
    """
    max_min_scaler=lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    # 归一化时间
    item_info_df['created_at_ts']=item_info_df[['created_at_ts']].apply(max_min_scaler)
    
    # 将click_article_id作为键，category_id作为值
    item_type_dict=dict(zip(item_info_df['click_article_id'],item_info_df['category_id']))
    # 将click_article_id作为键，words_count作为值
    item_words_dict=dict(zip(item_info_df['click_article_id'],item_info_df['words_count']))
    # 将click_article_id作为键，归一化后的created_at_ts作为值，创建一个字典
    item_created_time_dict=dict(zip(item_info_df['click_article_id'],item_info_df['created_at_ts']))
    
    # item_type_dict：文章ID到类别ID的映射
    # item_words_dict：文章ID到词数的映射
    # item_created_time_dict：文章ID到归一化创建时间的映射
    return item_type_dict, item_words_dict, item_created_time_dict


# 获取用户历史点击的文章信息
def get_user_hist_item_info_dict(all_click):
    
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs=all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict=dict(zip(user_hist_item_typs['user_id'],user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict=all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict=dict(zip(user_hist_item_ids_dict['user_id'],user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words=all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict=dict(zip(user_hist_item_words['user_id'],user_hist_item_words['words_count']))
    
    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict

# 获取点击次数topk个文章
def get_item_topk_click(click_df,k):
    topk_click=click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 召回数据打标签
def get_rank_label_df(recall_list_df, label_df, is_test):
    """
        :param recall_list_df: 每个user_id所召回的item，以及该item的分数
        :param label_df: 
        :param is_test: 是否有测试集
    """
    # 测试集是没有标签了，为了后面代码同一一些，这里直接给一个负数替代
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df
    
    label_df = label_df.rename(columns={'click_article_id': 'sim_item'})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'click_timestamp']], \
                                               how='left', on=['user_id', 'sim_item'])
    recall_list_df_['label'] = recall_list_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['click_timestamp']
    
    return recall_list_df_


# 获取召回列表
def get_user_recall_item_label_df(click_val,click_trn_hist, click_val_hist, click_tst_hist,click_trn_last, click_val_last, recall_list_df):
    # 获取训练数据的召回列表
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn_hist['user_id'].unique())]
    # 训练数据打标签
    trn_user_item_label_df = get_rank_label_df(trn_user_items_df, click_trn_last, is_test=False)
    # 训练数据负采样
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)
    
    if click_val is not None:
        val_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_val_hist['user_id'].unique())]
        val_user_item_label_df = get_rank_label_df(val_user_items_df, click_val_last, is_test=False)
        val_user_item_label_df = neg_sample_recall_data(val_user_item_label_df)
    else:
        val_user_item_label_df = None
        
    # 测试数据不需要进行负采样，直接对所有的召回商品进行打-1标签
    tst_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_tst_hist['user_id'].unique())]
    
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)
    
    return trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df

# 将召回数据转换为字典
# 将最终的召回的df数据转换成字典的形式做排序特征
def make_tuple_func(group_df):
    row_data=[]
    for name, row_df in group_df.iterrows():
        row_data.append((row_df['sim_item'],row_df['score'],row_df['label']))
    return row_data


# 节省内存的一个函数
# 减少内存
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df