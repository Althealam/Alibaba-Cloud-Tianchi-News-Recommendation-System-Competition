
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """
        召回效果的评估函数，主要通过计算命中率来评估推荐算法在不同召回数量下的效果
        : param user_recall_items_dict: 存储每个用户的召回物品列表，格式为{user_id:[(item_id,score),...]}
        : param trn_last_click_df: df，包含用户的最后一次点击记录，列名user_id和click_article_id分别表示用户ID和用户最后点击的物品ID
    """
    # 构建用户最后点击物品字典，键是user_id，值是用户最后点击的物品click_article_id
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    # 计算用户数量
    user_num = len(user_recall_items_dict)
    
    # 循环评估不同召回数量的效果
    for k in range(10, topk+1, 10):
        hit_num = 0 # 初始化命中数
        # 计算命中数和命中率
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)