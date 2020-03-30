import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.metrics import average_precision_score
import random
from tqdm import tqdm
from sklearn.metrics import average_precision_score,f1_score,precision_recall_curve,roc_auc_score,precision_recall_curve




n_classes = 5


def main():
    # df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_score.csv',index_col=0)
    # df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_score_fasterrcnn_inceptionv2_2000steps.csv',index_col=0)
    # df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_score_ssd_inceptionv2.csv',index_col=0)
    # df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_score_ssd_resnet50.csv',index_col=0)

    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_ssd_mobilenet_v1.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_ssd_mobilenet_v1_fpn.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_ssd_inception_v2.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_ssd_resnet50_fpn.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_faster_rcnn_inception_v2.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_faster_rcnn_resnet50.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_faster_rcnn_resnet101.csv',index_col=0)
    df_main = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/report/score/label_score_rfcn_resnet101.csv',index_col=0)

    ab_path = 'D:/Coding/SFU_CA/CMPT-733/groupproject/report/score_160k/'
    df_main = pd.read_csv(ab_path+'ssd_inception_v2_label_score.csv',index_col=0)
    df_main = pd.read_csv(ab_path+'ssd_mobilenet_v1_label_score.csv',index_col=0)


    print(df_main)
    y_score = df_main.iloc[:,:5].values
    y_test = df_main.iloc[:,5:].values

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i],average="macro")

    print(average_precision)


    precision["m"], recall["m"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision["m"] = average_precision_score(y_test, y_score,average="macro")
    print("Avg precision macro: ",average_precision['m'])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
    print("Avg precision micro: ",average_precision['micro'])

    precision["w"], recall["w"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision["w"] = average_precision_score(y_test, y_score,average="weighted")
    print("Avg precision weighted: ",average_precision['w'])
    # plt.clf()
    # plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    # plt.legend(loc="lower left")
    # plt.show()

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()
