import pandas as pd


def make_tsv():
    pola_result_path = './test/context_w9_pola_ckpt1040.txt'
    itst_result_path = './test/context_w1_itst_ckpt624.txt'
    emp_result_path = './test/context_w7_emp_ckpt624.txt'

    with open(pola_result_path, 'r') as f1:
        lines = f1.read().split('\n')[:-1]
        polas = [float(line.split(' ')[0]) for line in lines]

    with open(itst_result_path, 'r') as f2:
        lines = f2.read().split('\n')[:-1]
        itsts = [float(line.split(' ')[0]) for line in lines]

    with open(emp_result_path, 'r') as f3:
        lines = f3.read().split('\n')[:-1]
        emps = [float(line.split(' ')[0]) for line in lines]

    df = pd.DataFrame({'pola': polas, 'itst': itsts, 'emp': emps})
    df.to_csv('test_submit.tsv', header=False, index=False, sep='\t')


def trans_sequence():
    df = pd.read_csv('WASSA23_conv_with_labels_dev_sorted.csv')
    df3 = pd.read_csv('WASSA23_conv_level_dev.csv', encoding='gbk')
    df2 = pd.read_csv('test_submit.tsv', sep='\t', names=['pola', 'itst', 'emp'])
    df['pola_pred'] = df2['pola']
    df['itst_pred'] = df2['itst']
    df['emp_pred'] = df2['emp']
    df_result = pd.DataFrame()
    conv_ids = set(df['conversation_id'])
    for id in conv_ids:
        df_pred = df[df['conversation_id'] == id]
        df_pred.set_index('turn_id')
        df_ref = df3[df3['conversation_id'] == id]
        df_pred = df_pred.loc[df_ref['turn_id']]
        df_result = pd.concat([df_result, df_pred], axis=0)


def pred_dev():
    df = pd.read_csv('WASSA23_conv_with_labels_dev_sorted.csv')
    df2 = pd.read_csv('test_submit.tsv', sep='\t', names=['pola', 'itst', 'emp'])
    df['pola_pred'] = df2['pola']
    df['itst_pred'] = df2['itst']
    df['emp_pred'] = df2['emp']
    df.to_csv('wassa-dev-with-pred.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    df = pd.read_csv('wassa_test_final_wi_pred.csv', encoding='gbk')
    df_smt = df[['pola_pred', 'itst_pred', 'emp_pred']]
    df_smt.to_csv('test_final_smt1.tsv', sep='\t', header=False, index=False)



