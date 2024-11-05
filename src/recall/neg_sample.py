#### 负采样
# 负采样旨在减少负样本的数量，从而让模型聚焦于更重要的训练数据。
# 通常情况下，用户只会点击少量的推荐内容，大多数候选物品未被点击，因此被视为负样本
# 如果直接使用所有未点击的物品作为负样本，数据量会很庞大，导致训练成本增加并且模型可能受到负样本的强烈偏向影响
# 通过负采样减少负样本数量，能够提升训练效率和模型性能
#### 推荐算法中的负采样
# 1. 随机负采样： 从未点击物品中随机选择一部分作为负样本，简单且常用。
# 2. 基于特征的负采样：根据用户特征或物品特征来选择负样本。例如，可以优先选择与用户特征相似但未点击的物品作为负样本。
# 3. 基于流行度的负采样：优先采样热门或流行的物品作为负样本，增加难度，帮助模型更好地区分感兴趣和不感兴趣的物品。
import pandas as pd
from tqdm import tqdm


def recall_dict_2_df(recall_list_dict):
    """
        将召回的推荐列表转换为dataframe
        :param recall_list_dict: 字典，格式为{user_id:[(sim_item,score),...]}
        其中，user_id是用户id，sim_item是推荐的物品ID，score表示推荐的相似度得分
        return: recall_list_df 召回的dataframe，格式为[user_id,sim_item,score]
    """
    df_row_list = [] # [user, item, score]
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])
    
    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)
    
    return recall_list_df


def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    """
        实现负采样，以控制正负样本比例，降低推荐模型在负样本过多时的偏差
        函数可以给予用户和物品进行负采样，并且可以设置负样本的比例
        :param recall_items_df: 召回的物品dataframe
        :param sample_rate: 负采样的比例
    """
    # 分离正负样本
    # label=1表示正样本
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    # label=0表示负样本
    neg_data = recall_items_df[recall_items_df['label'] == 0]
    
    print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data)/len(neg_data))
    
    # 负样本采样函数
    def neg_sample_func(group_df):
        """
            分组采样函数
            对样本进行分组采样，限制每组采样的最大最小数量
            :param group_df: 需要进行分组采样的df
            return: group_df.sample(n=sample_num, replace=True) 分组采样后的df
        """
        # 计算采样数量
        neg_num = len(group_df)
        # 每个用户和物品的负样本数量由sample_rate控制
        sample_num = max(int(neg_num * sample_rate), 1) # 保证最少有一个
        sample_num = min(sample_num, 5) # 保证最多不超过5个，这里可以根据实际情况进行选择
        # 采样
        return group_df.sample(n=sample_num, replace=True)
    
    # 对用户进行负采样，保证所有用户都在采样后的数据中
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    # 对文章进行负采样，保证所有文章都在采样后的数据中
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)
    
    # 将上述两种情况下的采样数据合并
    neg_data_new = pd.concat([neg_data_user_sample,neg_data_item_sample])
    # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')
    
    # 将正样本数据合并
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)
    
    return data_new
