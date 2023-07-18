import os.path
import pickle
from collections import Counter

import pandas as pd
from scipy.stats import pearsonr
from tqdm import trange


def get_speakers():
    """
    get speaker info, write to a pickle file, a list of speaker dict.
    """

    pd.set_option('display.max_columns', None)

    speakers = []

    df = pd.read_excel('wassa-essay_train.xlsx')
    df = df[df['conversation_id'] >= 378]
    df = df.drop_duplicates(subset=['speaker_id'])
    for idx, line in df.iterrows():
        spk = {}
        if line['gender'] == 'unknown':
            continue
        spk['id'] = line['speaker_id']
        spk['gender'] = line['gender']
        spk['edu'] = line['education']
        spk['race'] = line['race']
        spk['age'] = line['age']
        spk['income'] = line['income']
        speakers.append(spk)
    with open('dev_speakers.pkl', 'wb') as f:
        pickle.dump(speakers, f)


def get_gender_split(speakers):
    """
        speakers: [{s1}, {s2}, xxx]
        @return: man speaker id list and women id list
        i don't know what 1 and 2 stand for
    """
    man_ids = []
    woman_ids = []
    for spk in speakers:
        if spk['gender'] == 1:
            man_ids.append(spk['id'])
        else:
            woman_ids.append(spk['id'])
    return man_ids, woman_ids


def get_race_split(speakers):
    """
        race 1 is one set, and 2-4 is the other
    """
    l1 = []
    l2 = []
    for spk in speakers:
        if spk['race'] in [2, 3, 4]:
            l2.append(spk['id'])
        else:
            l1.append(spk['id'])
    return l1, l2


def count_spk_dlg_num():
    pd.set_option('display.max_columns', None)

    result = {}

    df = pd.read_csv('essay.csv')
    print(df)
    for id in range(59, 100):
        temp = df[df['speaker_id'] == id]
        result[id] = len(temp)
    return result


def get_edu_split(speakers):
    """
    edu 2,3,4 is one set, the other...
    """
    l1 = []
    l2 = []
    for spk in speakers:
        if spk['edu'] in [2, 3, 4]:
            l1.append(spk['id'])
        else:
            l2.append(spk['id'])
    return l1, l2


def get_age_split(speakers):
    """
    age < 30 is one set, the other...
    """
    l1 = []
    l2 = []
    for spk in speakers:
        if spk['age'] < 30:
            l1.append(spk['id'])
        else:
            l2.append(spk['id'])
    return l1, l2


def get_income_split(speakers):
    """
    income < 35000 is one set, the other..
    """
    l1 = []
    l2 = []
    for spk in speakers:
        if spk['income'] < 35000:
            l1.append(spk['id'])
        else:
            l2.append(spk['id'])
    return l1, l2


def attribute_validate(attr, window_size, task, ckpt):
    """
    validate one checkpoint based on the split according to the attribution
    """
    task_to_label = {'pola': 'EmoPolarity', 'itst': 'EmoIntensity', 'emp': 'Empathy'}
    with open('speakers.pkl', 'rb') as f:
        speakers = pickle.load(f)
    if attr == 'race':
        l1, l2 = get_race_split(speakers)
    elif attr == 'gender':
        l1, l2 = get_gender_split(speakers)
    elif attr == 'edu':
        l1, l2 = get_edu_split(speakers)
    elif attr == 'age':
        l1, l2 = get_age_split(speakers)
    elif attr == 'income':
        l1, l2 = get_income_split(speakers)
    test_df = pd.read_csv('WASSA23_conv_with_labels_dev_sorted.csv')
    try:
        with open(f'test_xxl/context_w{window_size}_{task}_ckpt{ckpt}.txt', 'r') as f1:
            lines = f1.read().split('\n')[:-1]
            preds = [float(line.split(' ')[0]) for line in lines]  # get preds
            test_df['preds'] = preds
        df1 = test_df[test_df['speaker_id'].isin(l1)]
        df2 = test_df[test_df['speaker_id'].isin(l2)]
        pear1 = pearsonr(df1[task_to_label[task]], df1['preds']).statistic
        pear2 = pearsonr(df2[task_to_label[task]], df2['preds']).statistic
        return pear1, pear2
    except FileNotFoundError:
        return None


def subset_validate(speaker_list, window_size, task, ckpt):
    """
    validate one checkpoint on a speaker list.
    @return: a pearson coef.
    """
    task_to_label = {'pola': 'EmoPolarity', 'itst': 'EmoIntensity', 'emp': 'Empathy'}
    test_df = pd.read_csv('WASSA23_conv_with_labels_my_dev_sorted.csv', encoding='gbk')
    try:
        with open(f'dev_xxl/context_w{window_size}_{task}_ckpt{ckpt}.txt', 'r') as f1:
            lines = f1.read().split('\n')[:-1]
            preds = [float(line.split(' ')[0]) for line in lines]  # get preds
            test_df['preds'] = preds
        df1 = test_df[test_df['speaker_id'].isin(speaker_list)]
        if len(df1) == 0:
            return -1
        pear = pearsonr(df1[task_to_label[task]], df1['preds']).statistic
        return pear
    except FileNotFoundError:
        return None


def select_speakers(gender, education, race, age, income):
    """
    @return: a List, based on the attributions given.
    """
    with open('dev_speakers.pkl', 'rb') as f:
        speakers = pickle.load(f)

    spk_list = []
    for spk in speakers:
        if education == 'high':  # 5-7
            if spk['edu'] < 5 :
                continue
        elif education == 'low':  # 2-4
            if spk['edu'] > 4:
                continue

        if age == 'young':
            if spk['age'] >= 30:
                continue
        elif age == 'old':
            if spk['age'] < 30:
                continue

        if income == 'high':
            if spk['income'] < 35000:
                continue
        elif income == 'low':
            if spk['income'] >= 35000:
                continue

        if spk['gender'] == gender and spk['race'] == race:
            spk_list.append(spk['id'])

    return spk_list


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # gender = 1
    # education = 'high'  # or low
    # race = 1  # or rare
    # age = 'young'  # or old
    # income = 'high'  # or low
    #
    # spk_list = select_speakers(gender, education, race, age, income)
    # print(len(spk_list))
    # print(spk_list)

    # ws_list = [1, 3, 5, 7, 9, 11]
    # ckpt_list = [208 * i for i in range(1, 7)]
    # name_list = [None]
    # p1_list = []
    #
    # result_dict = {}
    #
    # df = pd.read_csv('WASSA23_conv_with_labels_my_dev_sorted.csv', encoding='gbk')
    #
    # for gender in [1, 2]:
    #     for race in [1, 2, 3, 5]:
    #         for edu in ['high', 'low']:
    #             for age in ['young', 'old']:
    #                 for income in ['high', 'low']:
    #                     result_dict[f'g{gender}_r{race}_e{edu}_a{age}_i{income}'] = list()
    #                     result_dict[f'g{gender}_r{race}_e{edu}_a{age}_i{income}'].append(
    #                         len(df[df['speaker_id'].isin(select_speakers(gender, edu, race, age, income))])
    #                     )
    #
    # for tsk in ['pola', 'itst', 'emp']:
    #     for ws in trange(1, 13, 2):
    #         for ckpt in ckpt_list: # one checkpoint
    #             for gender in [1,2]:
    #                 for race in [1,2,3,5]:
    #                     for edu in ['high', 'low']:
    #                         for age in ['young', 'old']:
    #                             for income in ['high', 'low']:
    #                                 spk_list = select_speakers(gender, edu, race, age, income)
    #                                 if spk_list:
    #                                     pear = subset_validate(spk_list, ws, tsk, ckpt)
    #                                 else:
    #                                     pear = -1
    #                                 result_dict[f'g{gender}_r{race}_e{edu}_a{age}_i{income}'].append(pear)
    #             name_list.append(f'w{ws}_{tsk}_ckpt{ckpt}')
    # result_dict['model_name'] = name_list
    # df = pd.DataFrame(result_dict)
    # with open('dev_xxl_df.pkl', 'wb') as f2:
    #     pickle.dump(df, f2)
    #     # df = pickle.load(f2)
    # df.to_csv('temp.csv', index=False)

    base = 'dev_xxl'

    ws_list = [1, 3, 5, 7, 9, 11]
    task_list = ['pola', 'itst', 'emp']
    ckpt_list = [208 * i for i in range(1, 7)]
    name_list = []
    pear_list = []
    for tsk in task_list:
        for ws in ws_list:
            for ckpt in ckpt_list:
                fname = f'context_w{ws}_{tsk}_ckpt{ckpt}.txt'
                path = os.path.join(base, fname)
                try:
                    with open(path, 'r') as f1:
                        pear = float(f1.read().split('\n')[-1].split(' ')[-1])
                except FileNotFoundError:
                    continue
                name_list.append(fname[:-4])
                pear_list.append(pear)
    df = pd.DataFrame({'model_name': name_list, 'Test_Pearson': pear_list})
    df.to_csv('temp.csv', index=False)
