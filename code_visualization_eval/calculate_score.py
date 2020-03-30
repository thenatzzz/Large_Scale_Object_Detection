import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.metrics import average_precision_score
import random
from tqdm import tqdm
from sklearn.metrics import average_precision_score,f1_score,precision_recall_curve,roc_auc_score,precision_recall_curve


N_CLASSES = 5
LIST_NEGATIVE_SAMPLES = [50000,100000,150000]
# KEYS =['Model','Gun','Knife','Wrench','Pliers','Scissors','mean Avg Precision(Macro)','mean Avg Precision(Micro)','mean Avg Precision(Weighted)']
KEYS =['Model','Gun','Knife','Wrench','Pliers','Scissors','Macro mAP','Micro mAP','Weighted mAP']

def calculate_score(model_name,df_positive,df_negative):
    average_precision = {key: [] for key in KEYS}
    for neg_sample in LIST_NEGATIVE_SAMPLES:
        df_combined = pd.DataFrame()
        df_negative_temp =df_negative.sample(n=neg_sample,random_state=1)
        df_combined = pd.concat([df_positive, df_negative_temp], ignore_index=True, sort =False)
        y_score = df_combined.iloc[:,:5].values
        y_test = df_combined.iloc[:,5:].values
        for i in range(N_CLASSES):
            average_precision[KEYS[i+1]].append(average_precision_score(y_test[:, i], y_score[:, i],average="macro"))
        average_precision[KEYS[-3]].append(average_precision_score(y_test, y_score,average="macro"))
        average_precision[KEYS[-2]].append(average_precision_score(y_test, y_score,average="micro"))
        average_precision[KEYS[-1]].append(average_precision_score(y_test, y_score,average="weighted"))
        average_precision[KEYS[0]] = model_name
    return average_precision
def get_starting_idx_positive_sample(df):
    index_pos = df.index.str.contains(r'P').sum()
    return index_pos
def get_list_of_files(path):
    list_files = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files
def strip_file_output(elem):
    temp_len = len('_label_score.csv')
    return elem[:len(elem)-temp_len]

def main():
    absolute_path = 'D:/Coding/SFU_CA/CMPT-733/groupproject/report/score_160k/'
    list_models = get_list_of_files(absolute_path)
    print(list_models)
    list_dicts =[]
    for model in list_models:
        model_name = strip_file_output(model)
        print(model_name)
        df = pd.read_csv(absolute_path+model,index_col=0)
        df_main = df.sort_values(by=['filename'], ascending=False)
        index_pos = get_starting_idx_positive_sample(df_main)
        df_positive = df_main.iloc[:index_pos,:].copy()
        df_negative = df_main.iloc[index_pos:,:].copy()

        list_dicts.append(calculate_score(model_name,df_positive,df_negative))
        print(df_main,'\n')
    big_list =[]
    for dict in list_dicts:
        temp_list = []
        for key,val in dict.items():
            if type(val) is not str:
                val = [ elem*100 for elem in val ]
                val = [ '%.3f' % elem for elem in val ]
                print(val)
                temp_list = temp_list+val
            else:
                temp_list = temp_list+[val]
        big_list.append(temp_list)

    col_names =[]
    col_names.append(KEYS[0])
    suffix = ['(50k)','(100k)','(150k)']
    idx=0
    for key in KEYS[1:]:
        col_names.append(key+suffix[0])
        col_names.append(key+suffix[1])
        col_names.append(key+suffix[2])
        idx += 1
    new_df = pd.DataFrame.from_records(big_list,columns=col_names)
    print(new_df)
    new_df.to_csv('model_scores.csv',index=None)


if __name__ == '__main__':
    main()
