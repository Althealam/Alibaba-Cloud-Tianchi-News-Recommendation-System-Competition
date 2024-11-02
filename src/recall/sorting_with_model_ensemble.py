import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
# 导入deepctr
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import * 
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



# 返回排序后的结果
def submit(recall_df,topk=5, model_name=None):
    recall_df=recall_df.sort_values(by=['user_id','pred_score'])
    recall_df['rank']=recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False,method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp=recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min()>=topk

    del recall_df['pred_score']
    submit=recall_df[recall_df['rank']<=topk].set_index(['user_id','rank']).unstack(-1).reset_index()

    submit.columns=[int(col) if isinstance(col,int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit=submit.rename(columns={'':'user_id',1:'article_1',2:'article_2',3:'article_3',4:'article_4',5:'article_5'})

    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    save_name=save_path+model_name+'_'+datetime.today().strftime('%m-%d')+'.csv'
    submit.to_csv(save_name,index=False,header=True)


# 排序结果归一化
def norm_sim(sim_df,weight=0.0):
    min_sim=sim_df.min()
    max_sim=sim_df.max()
    if max_sim==min_sim:
        sim_df=sim_df.apply(lambda sim:1.0)
    else:
        sim_df=sim_df.apply(lambda sim: 1.0*(sim-min_sim)/(max_sim-min_sim))
    
    sim_df=sim_df.apply(lambda sim:sim+weight)
    return sim_df


# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, his_behavior_fea, emb_dim=32, max_len=100):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    his_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度， 这里为了简单， 统一把离散型特征列采用一样的隐向量维度
    max_len: 用户序列的最大长度
    """
    
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique() + 1, embedding_dim=emb_dim) for feat in sparse_fea]
    
    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_fea]
    
    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=df['click_article_id'].nunique() + 1,
                                    embedding_dim=emb_dim, embedding_name='click_article_id'), maxlen=max_len) for feat in his_behavior_fea]
    
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    
    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in his_behavior_fea:
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')      # 二维数组
        else:
            x[name] = df[name].values
    
    return x, dnn_feature_columns


# 模型融合
# 1. 加权融合
def get_ensemble_predict_topk(rank_model, topk=5):
    final_recall = rank_model['lgb_cls'].append(rank_model['din_ranker'])
    rank_model['lgb_ranker']['pred_score'] = rank_model['lgb_ranker']['pred_score'].transform(lambda x: norm_sim(x))
    
    final_recall = final_recall.append(rank_model['lgb_ranker'])
    final_recall = final_recall.groupby(['user_id', 'click_article_id'])['pred_score'].sum().reset_index()
    
    submit(final_recall, topk=topk, model_name='ensemble_fuse')