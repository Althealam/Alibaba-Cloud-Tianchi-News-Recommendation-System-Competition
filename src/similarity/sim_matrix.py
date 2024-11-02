from config import get_user_item_time,get_item_user_time_dict
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import math, pickle
from sklearn.preprocessing import MinMaxScaler
import faiss
import collections


# 基于物品的协同过滤
def itemcf_sim(df,item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算-->基于物品的协同过滤
        df:数据表
        item_created_time_dict:文章创建时间的字典
        return: 文章与文章的相似性矩阵
    """
    user_item_time_dict=get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim=defaultdict(dict) # 存储文章之间的相似度
    item_cnt=defaultdict(int) # 记录每个文章被点击的次数
    
    for user,item_time_list in tqdm(user_item_time_dict.items(),total=len(user_item_time_dict)):
        # user: 用户ID
        # item_time_list: 该用户点击的文章和时间的列表
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i,i_click_time) in enumerate(item_time_list):
        # 遍历用户点击的每篇文章i和点击时间i_click_time
        
            item_cnt[i]+=1 # 更新文章i的点击次数
            i2i_sim.setdefault(i,{}) # 确保i2i_sim字典中存在文章i的键
            
            for loc2,(j,j_click_time) in enumerate(item_time_list):
                # 遍历用户点击的文章列表，用于计算i和j之间的相似度
                if i==j:
                    # 如果文章i和j是同一篇文章，则跳过计算
                    continue
                
                #### 1. 用户点击的顺序权重 ####
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha=1.0 if loc2>loc1 else 0.7 # 文章在用户点击序列中的位置设置权重loc_alpha
                
                # 位置信息权重，其中的参数可以调节（考虑了文章在用户点击序列中的距离）
                loc_weight=loc_alpha*(0.9**(np.abs(loc2-loc1)-1))
                
                #### 2. 用户点击的时间权重 ####
                # 点击时间权重，其中的参数可以调节（考虑了文章被点击的时间差异）
                click_time_weight=np.exp(0.7**np.abs(i_click_time-j_click_time))
                
                #### 3. 文章创建的时间权重 ####
                # 两篇文章创建时间的权重，其中的参数可以调节（考虑了文章发布时间的差异）
                created_time_weight=np.exp(0.8**np.abs(item_created_time_dict[i]-item_created_time_dict[j]))
                
                i2i_sim[i].setdefault(j,0) # 确保i2i_sim字典中存在文章j的键
                # 更新文章i和文章j之间的相似度
                i2i_sim[i][j]+=loc_weight*click_time_weight*created_time_weight/math.log(len(item_time_list)+1)
                
    i2i_sim_ = {i: dict(i2i_sim[i]) for i in i2i_sim}
    for i, related_items in i2i_sim.items():
        for j,wij in related_items.items():
            i2i_sim_[i][j]=wij/math.sqrt(item_cnt[i]*item_cnt[j])
        
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_,open(save_path+'itemcf_i2i_sim.pkl','wb'))
        
    return i2i_sim_


# item embedding sim 
def embedding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于文章的嵌入向量embedding，计算文章之间的相似性矩阵
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵
        
        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    
    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
    
    # 提取嵌入向量，并将向量转换为NumPy数组，并确保其为连续数组
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    
    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk) # 返回的是列表
    
    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    # rele_idx_list是通过FAISS搜索得到的与目标文章嵌入向量最相似的文章索引列表
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]): 
            rele_raw_id = item_idx_2_rawid_dict.get(rele_idx)
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
    
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))   
    
    return item_sim_dict


# 基于用户的协同过滤
def get_user_activate_degree_dict(all_click_df):
    all_click_df_=all_click_df.groupby('user_id')['click_article_id'].count().reset_index()
    
    # 用户活跃度归一化
    mm=MinMaxScaler()
    all_click_df_['click_article_id']=mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict=dict(zip(all_click_df_['user_id'],all_click_df_['click_article_id']))
    
    return user_activate_degree_dict

def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        all_click_df: 数据表
        user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵
        
        思路：基于用户的协同过滤+关联规则
    """
    item_user_time_dict=get_item_user_time_dict(all_click_df)
    
    u2u_sim={}
    user_cnt=defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u]+=1
            u2u_sim.setdefault(u,{})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v,0)
                if u==v:
                    continue
                # 用户平均活跃度作为活跃度的权重
                activate_weight=100*0.5*(user_activate_degree_dict[u]+user_activate_degree_dict[v])
                u2u_sim[u][v]+=activate_weight/math.log(len(user_time_list)+1)
    u2u_sim_=u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v,wij in related_users.items():
            u2u_sim_[u][v]=wij/math.sqrt(user_cnt[u]*user_cnt[v])
    
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    # 得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_,open(save_path+'usercf_u2u_sim.pkl','wb'))
    
    return u2u_sim_
