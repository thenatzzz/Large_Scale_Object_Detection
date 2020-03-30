''' Splitting Xray csv file(xml object position) into Xray csv train/test file '''

import os
import glob
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_list_of_files(path):
    list_files = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files

def create_csv_label(list_files,df_label,save_path_filename):
    df_new = df_label[df_label['filename'].isin(list_files)]
    df_new.to_csv(save_path_filename,index=None)

def main():
    # path =image_path = os.path.join(os.getcwd(), 'dataset/new_dataset/positive_test')
    path =image_path = os.path.join(os.getcwd(), 'dataset/new_dataset/positive_train')
    list_img = get_list_of_files(path)

    df_loc_class = pd.read_csv('xray.csv')
    # create_csv_label(list_img,df_loc_class,'xray_positive_test.csv')
    create_csv_label(list_img,df_loc_class,'xray_positive_train.csv')

if __name__ == '__main__':
    main()
