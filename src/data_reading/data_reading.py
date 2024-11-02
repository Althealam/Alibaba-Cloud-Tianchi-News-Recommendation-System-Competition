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
import logging
from gensim.models import Word2Vec

# debug模式：从训练集中划分一些数据来调试代码
def get_all_click_sample(data_path, sample_nums):
    """
        从训练数据集中随机抽取一部分数据用于调试
        data_path：原数据的存储路径
        sample_nums：采样数目
    """
    all_click=pd.read_csv(data_path+'train_click_log.csv')
    all_user_ids=all_click.user_id.unique()
    
    # 随机抽取sample_nums个数据
    sample_user_ids=np.random.choice(all_user_ids,size=sample_nums,replace=False)
    # 获取sample_user_ids对应的在all_click中的数据
    all_click=all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click=all_click.drop_duplicates((['user_id','click_article_id','click_timestamp']))
    return all_click

# 读取点击数据
def get_all_click_df(data_path,offline=False):
    """
        从给定的路径中读取点击数据，并根据offline参数决定是仅读取训练数据还是同时读取训练和测试数据
        data_path：原数据的存储路径
        offline：表示是否处于离线模式。在离线模式下，只处理训练数据，否则，同时处理训练和测试数据
    """
    if offline:
        all_click=pd.read_csv(data_path+'train_click_log.csv')
    else:
        trn_click=pd.read_csv(data_path+'train_click_log.csv')
        tst_click=pd.read_csv(data_path+'testA_click_log.csv')
        
        # 包含测试集和训练集
        all_click=pd.concat([trn_click,tst_click])
    
    # 去除重复的点击记录，保留唯一的(user_id, click_article_id, click_timestamp)组合
    all_click=all_click.drop_duplicates((['user_id','click_article_id','click_timestamp']))
    return all_click

# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df=pd.read_csv(data_path+'articles.csv')
    
    # 为了与训练集中的click_article_id进行拼接，修改article_id为click_article_id
    item_info_df=item_info_df.rename(columns={'article_id':'click_article_id'})
    
    return item_info_df

# 读取文章的embedding属性
def get_item_emb_dict(data_path):
    item_emb_df=pd.read_csv(data_path+'articles_emb.csv')
    # 创建列表item_emb_cols，包含item_emb_df中所有列名包含'emb'的列（用于筛选出包含嵌入向量的列）
    item_emb_cols=[x for x in item_emb_df.columns if 'emb' in x]
    # 利用ascontiguousarray函数将筛选出的嵌入向量列转换为一个连续的Numpy数组item_emb_np
    item_emb_np=np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np=item_emb_np/np.linalg.norm(item_emb_np,axis=1,keepdims=True)
    
    # 创建字典，将item_emb_df中的article_id列的值作为字典的键，将对应的归一化嵌入向量作为字典的值
    item_emb_dict=dict(zip(item_emb_df['article_id'],item_emb_np))
    # 使用pickle库将item_emb_dict字典序列化并保存到文件中
    # wb表示以二进制写入模式打开文件
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    pickle.dump(item_emb_dict,open(save_path+'item_content_emb.pkl','wb'))
    
    return item_emb_dict

# 节省内存函数
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

# 划分验证集和训练集
# all_click_df：训练集
# sample_user_nums：验证集的用户数量
def trn_val_split(all_click_df,sample_user_nums):
    """
        目的：从给定的训练集数据中分割出验证集和验证集的答案（验证集的标签）
        输入：all_click_df：完整的训练集数据，包含用户点击文章的记录
             sample_user_nums：需要抽取作为验证集的用户数量
        输出：click_trn：训练集数据
             click_val：验证集数据
             val_ans：验证集的答案，即验证集中每个用户的最后一次点击
    """
    # 初始化
    all_click=all_click_df
    all_user_ids=all_click.user_id.unique()
    
    # 用户抽样（随机选择sample_user_nums个用户ID作为验证集用户）
    # replace=True表示可以重复抽样，反之不可以
    sample_user_ids=np.random.choice(all_user_ids,size=sample_user_nums,replace=False)
    
    # 分割训练集和验证集
    click_val=all_click[all_click['user_id'].isin(sample_user_ids)] # 验证集用户点击数据
    click_trn=all_click[~all_click['user_id'].isin(sample_user_ids)] # 训练集用户点击数据
    
    # 排序和提取验证集答案
    click_val=click_val.sort_values(['user_id','click_timestamp'])
    val_ans=click_val.groupby('user_id').tail(1) # 提取最后一次点击作为答案
    
    # 清理验证集（去除答案数据，确保答案和验证集对应）
    # 移除每个用户在验证集中的最后一次点击
    click_val=click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)
    # 确保答案数据集中只包含那些在清理后的验证集中存在的用户
    val_ans=val_ans[val_ans.user_id.isin(click_val.user_id.unique())]
    # 进一步确保清理后的验证集中也只包含那些在答案数据集中存在的用户
    click_val=click_val[click_val.user_id.isin(val_ans.user_id.unique())]
    
    return click_trn, click_val, val_ans
    # 训练集数据：click_trn
    # 验证集数据：click_val
    # 验证集答案：val_ans

# 读取训练、验证以及测试集
def get_trn_val_tst_data(data_path,offline=False):
    if offline:
        click_trn_data = pd.read_csv(data_path+'train_click_log.csv')  # 训练集用户点击日志
        # click_trn_data=all_click_df
        click_trn_data = reduce_mem(click_trn_data)
        click_trn, click_val, val_ans = trn_val_split(click_trn_data, sample_user_nums)
    else:
        click_trn = pd.read_csv(data_path+'train_click_log.csv')
        click_trn = reduce_mem(click_trn)
        click_val = None
        val_ans = None
    
    click_tst = pd.read_csv(data_path+'testA_click_log.csv')
    
    return click_trn, click_val, click_tst, val_ans

# 返回多路召回列表或者单路召回
def get_recall_list(save_path, single_recall_model=None, multi_recall=False):
    # 多路召回
    if multi_recall:
        return pickle.load(open(save_path + 'final_recall_items_dict.pkl', 'rb'))
    
    # 单路召回
    # 1. 基于文章的协同过滤
    if single_recall_model == 'i2i_itemcf':
        return pickle.load(open(save_path + 'itemcf_recall_dict.pkl', 'rb'))
    # 2. 基于文章embedding
    elif single_recall_model == 'i2i_emb_itemcf':
        return pickle.load(open(save_path + 'itemcf_emb_dict.pkl', 'rb'))
    

# 读取各种embedding
def train_itrem_word2vec(click_df,embed_size=64,save_name='item_w2v_emb.pkl',split_char=' '):
    click_df=click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id']=click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs=click_df.groupby(['user_id'])['click_article_id'].apply(lambda x:list(x)).reset_index()
    docs=docs['click_article_id'].values.tolist()
    
    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
    
    w2v=Word2Vec(docs,size=16,sg=1,window=5,seed=2020,workers=24,min_count=1,iter=1)
    
    # 保存成字典的形式
    item_w2v_emb_dict={k: w2v[k] for k in click_df['click_article_id']}
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp'
    pickle.dump(item_w2v_emb_dict,open(save_path+'item_w2v_emb.pkl','wb'))
    
    return item_w2v_emb_dict


# 可以通过字典查询对应的item的Embedding
def get_embedding(save_path, all_click_df):
    if os.path.exists(save_path + 'item_content_emb.pkl'):
        item_content_emb_dict = pickle.load(open(save_path + 'item_content_emb.pkl', 'rb'))
    else:
        print('item_content_emb.pkl 文件不存在...')
        
    # w2v Embedding是需要提前训练好的
    if os.path.exists(save_path + 'item_w2v_emb.pkl'):
        item_w2v_emb_dict = pickle.load(open(save_path + 'item_w2v_emb.pkl', 'rb'))
    else:
        item_w2v_emb_dict = trian_item_word2vec(all_click_df)
        
    if os.path.exists(save_path + 'item_youtube_emb.pkl'):
        item_youtube_emb_dict = pickle.load(open(save_path + 'item_youtube_emb.pkl', 'rb'))
    else:
        print('item_youtube_emb.pkl 文件不存在...')
    
    if os.path.exists(save_path + 'user_youtube_emb.pkl'):
        user_youtube_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))
    else:
        print('user_youtube_emb.pkl 文件不存在...')
    
    return item_content_emb_dict, item_w2v_emb_dict, item_youtube_emb_dict, user_youtube_emb_dict

# 读取文章信息
def get_article_info_df():
    article_info_df=pd.read_csv('/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/data/articles.csv')
    article_info_df = reduce_mem(article_info_df)
    
    return article_info_df


# 读取各种embedding
def trian_item_word2vec(click_df, embed_size=64, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符串才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转换成句子的形式
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()

    # 为了方便查看训练的进度，这里设定一个log信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # 这里的参数对训练得到的向量影响也很大,默认负采样为5
    w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=24, min_count=1,epochs=1)
    
    # 保存成字典的形式，并且保存到item_w2v_emb.pkl中
    item_w2v_emb_dict = {k: w2v.wv[k] for k in click_df['click_article_id']}
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp'
    pickle.dump(item_w2v_emb_dict, open(save_path + 'item_w2v_emb.pkl', 'wb'))
    
    return item_w2v_emb_dict


# 可以通过字典查询对应的item的Embedding
def get_embedding(save_path, all_click_df):
    if os.path.exists(save_path + 'item_content_emb.pkl'):
        item_content_emb_dict = pickle.load(open(save_path + 'item_content_emb.pkl', 'rb'))
    else:
        print('item_content_emb.pkl 文件不存在...')
        
    # w2v Embedding是需要提前训练好的
    if os.path.exists(save_path + 'item_w2v_emb.pkl'):
        item_w2v_emb_dict = pickle.load(open(save_path + 'item_w2v_emb.pkl', 'rb'))
    else:
        item_w2v_emb_dict = trian_item_word2vec(all_click_df)
    
    return item_content_emb_dict, item_w2v_emb_dict
