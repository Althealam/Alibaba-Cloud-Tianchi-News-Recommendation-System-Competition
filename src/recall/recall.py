import pickle
import collections
import faiss
import numpy as np
from tqdm import tqdm
import os

# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵
        
        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
            
            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc))
            
            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]
                
            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij
    
    # 不足10个，用热门文章补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank



# 基于用户的召回u2u
def user_based_recommend(user_id,user_item_time_dict,u2u_sim,sim_user_topk,recall_item_num,item_topk_click,item_created_time_dict,emb_i2i_sim):
    """
        基于文章协同过滤的召回
        user_id: 用户id
        user_item_time_dict: 字典，键是用户ID，值是用户点击过的文章ID和点击时间的列表，根据点击时间获取用户的点击文章序列 {user1:{item1:time1,item2:time2,...}}
        u2u_sim: 字典，文章相似性矩阵，键是用户ID，值是与其他用户相似度的字典
        sim_user_topk: 整数，选择与当前用户最相似的前k个用户
        recall_item_num: 整数，最后的召回文章数量
        item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        item_created_time_click: 字典，文章创建时间列表，键是文章ID，值是文章的创建时间
        emb_i2i_sim: 字典，基于内容embedding算的文章相似性矩阵

        return：召回的文章列表 {item1:score1,item2:score2,...}
    """
    # 获取用户历史交互的文章
    user_item_time_list=user_item_time_dict[user_id]
    user_hist_items=set([i for i,t in user_item_time_list]) # 存在一个用户与某篇文章的多次交互

    # 初始化召回文章的得分字典
    items_rank={}
    # 遍历相似用户的文章（与目标用户相似度最高的sim_user_topk个用户）
    for sim_u, wuv in sorted(u2u_sim[user_id].items(),key=lambda x:x[1],reverse=True)[:sim_user_topk]:
        """
            sim_u: 相似用户
            wuv: 相似用户权重（用户u与用户v的相似度）
        """
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i,0)

            loc_weight=1.0 # 点击时的相似位置权重（这里简单的将所有位置的权重设置为1）
            content_weight=1.0 # 内容相似性权重，如果文章i与目标用户历史点击的文章j在内容上相似，则增加权重
            created_time_weight=1.0 # 创建时间差权重（时间差越大，权重越小）

            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc,(j,click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight+=0.9**(len(user_item_time_list)-loc)
                
                # 内容相似性权重
                # 如果文章i与目标用户历史点击的文章j在内容上相似，则增加权重
                if emb_i2i_sim.get(i,{}).get(j,None) is not None:
                    content_weight+=emb_i2i_sim[i][j]
     
                if emb_i2i_sim.get(j,{}).get(i,None) is not None:
                    content_weight+=emb_i2i_sim[j][i]


                # 创建时间差权重
                created_time_weight+=np.exp(0.8*np.abs(item_created_time_dict[i]-item_created_time_dict[j]))

            items_rank[i]+=loc_weight*content_weight*created_time_weight*wuv

    # 热度补全
    if len(items_rank)<recall_item_num:
        for i,item in enumerate(item_topk_click):
            if item in items_rank.items():
                continue
            items_rank[item]=-i-100 
            if len(items_rank)==recall_item_num:
                break
                    
    # 对召回文章进行排序并返回（对召回文章按照得分进行降序排序，并且返回得分最高的前recall_item_num篇文章）
    items_rank=sorted(items_rank.items(),key=lambda x:x[1],reverse=True)[:recall_item_num]

    return items_rank


# 多路召回合并
def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}
    
    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list
        
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        
        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
            
        return norm_sorted_item_list
    
    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]
        
        for user_id, sorted_item_list in user_recall_items.items(): # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)
        
        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score  
    
    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))

    return final_recall_items_dict_rank