''' Main Visualization file:
2 figues:
1. mean Average Precision curves of different models
2. mean Average Precision (micro) of different models and different ratio of positive-negative test dataset
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score,f1_score,precision_recall_curve,roc_auc_score,precision_recall_curve
from os import listdir
from os.path import isfile, join

# Index(['Model', 'Gun(50k)', 'Gun(100k)', 'Gun(150k)', 'Knife(50k)',
#        'Knife(100k)', 'Knife(150k)', 'Wrench(50k)', 'Wrench(100k)',
#        'Wrench(150k)', 'Pliers(50k)', 'Pliers(100k)', 'Pliers(150k)',
#        'Scissors(50k)', 'Scissors(100k)', 'Scissors(150k)', 'Macro mAP(50k)',
#        'Macro mAP(100k)', 'Macro mAP(150k)', 'Micro mAP(50k)',
#        'Micro mAP(100k)', 'Micro mAP(150k)', 'Weighted mAP(50k)',
#        'Weighted mAP(100k)', 'Weighted mAP(150k)'],
N_CLASSES=5
list_color = ['y','g','r','m','b','salmon','gray','c']

def plot_score(start_idx,df):
    x = ['50k negative', '100k negative', '150k negative']
    fig = plt.figure()
    ax = plt.subplot(111)

    for each_model in range(df.shape[0]):
        y= df.iloc[each_model,start_idx:start_idx+3]
        plt.plot(x, y,linestyle='--', marker='o', color=list_color[each_model],label=df.iloc[each_model,0])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])

    plt.xlabel('Number of negative samples per 1,800 positive samples',fontweight='bold')
    plt.ylabel('Accuracy (mean average precision)',fontweight='bold')
    plt.title('Detection',fontweight='bold')
    plt.savefig('detection_scores.png')

    plt.show()
def get_list_of_files(path):
    list_files = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files
def strip_file_output(elem):
    temp_len = len('_label_score.csv')
    return elem[:len(elem)-temp_len]
def plot_precision_recall_curve():
    fig, ax = plt.subplots()

    ab_path = 'D:/Coding/SFU_CA/CMPT-733/groupproject/report/score_160k/'
    list_model_files = get_list_of_files(ab_path)
    list_model_names = [ strip_file_output(model) for model in list_model_files]

    precision = dict()
    recall = dict()
    average_precision = dict()

    plt.clf()
    plt.figure(figsize=(7,6))
    count =0
    for model_name,model_file in zip(list_model_names,list_model_files):
        df_main = pd.read_csv(ab_path+model_file,index_col=0)
        y_score = df_main.iloc[:,:5].values
        y_test = df_main.iloc[:,5:].values
        # Compute micro-average ROC curve and ROC area
        precision["micro"+'_'+model_name], recall["micro"+'_'+model_name], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
        average_precision["micro"+'_'+model_name] = average_precision_score(y_test, y_score,average="micro")
        plt.plot(recall["micro"+'_'+model_name], precision["micro"+'_'+model_name],color=list_color[count],
            label='{0} (area= {1:0.2f})'''.format(model_name,average_precision["micro"+'_'+model_name]))
        count += 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall',fontweight='bold')
    plt.ylabel('Precision',fontweight='bold')
    plt.title('Precision-Recall curve for Object Detection models',fontweight='bold')
    plt.legend(loc="lower left")
    plt.savefig('avg_precision_recall.png')
    plt.show()

def plot_precision_recall_curve2(neg_sample):
    fig, ax = plt.subplots()

    ab_path = 'D:/Coding/SFU_CA/CMPT-733/groupproject/report/score_160k/'
    list_model_files = get_list_of_files(ab_path)
    list_model_names = [ strip_file_output(model) for model in list_model_files]

    precision = dict()
    recall = dict()
    average_precision = dict()

    plt.clf()
    plt.figure(figsize=(7,6))
    count =0
    for model_name,model_file in zip(list_model_names,list_model_files):
        df_main = pd.read_csv(ab_path+model_file,index_col=0)
        df_main = df_main.sort_index(ascending=False)

        df_pos = df_main.iloc[:1829,:]
        df_neg = df_main.iloc[1829:,:]
        df_neg =df_neg.sample(n=neg_sample,random_state=1)
        df_combined = pd.concat([df_pos, df_neg], ignore_index=True, sort =False)
        print(df_combined)

        y_score = df_combined.iloc[:,:5].values
        y_test = df_combined.iloc[:,5:].values
        # Compute micro-average ROC curve and ROC area
        precision["micro"+'_'+model_name], recall["micro"+'_'+model_name], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
        average_precision["micro"+'_'+model_name] = average_precision_score(y_test, y_score,average="micro")
        plt.plot(recall["micro"+'_'+model_name], precision["micro"+'_'+model_name],color=list_color[count],
            label='{0} (area= {1:0.2f})'''.format(model_name,average_precision["micro"+'_'+model_name]))
        count += 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall',fontweight='bold')
    plt.ylabel('Precision',fontweight='bold')
    # plt.title('Precision-Recall curve (50k Negative samples)',fontweight='bold')
    # plt.title('Precision-Recall curve (100k Negative samples)',fontweight='bold')
    plt.title('Precision-Recall curve (150k Negative samples)',fontweight='bold')

    plt.legend(loc="lower left")
    # plt.savefig('avg_precision_recall_50k.png')
    # plt.savefig('avg_precision_recall_100k.png')
    plt.savefig('avg_precision_recall_150k.png')
    plt.show()

def main():
    df = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject_git/report/main/scores.csv')
    print(df)
    dict_str_idx = {'gun':1,'knife':4,'wrench':7,'pliers':10,'scissors':13,'macro_map':16,'micro_map':19,'weighted_map':22}

    # plot_score(dict_str_idx['micro_map'],df)
    # plot_precision_recall_curve()
    # plot_precision_recall_curve2(150000)
if __name__ == '__main__':
    main()
