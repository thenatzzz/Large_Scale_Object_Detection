'''Combine all label csv files according to selected folders of images into final csv for training/testing '''

import os
import glob
import pandas as pd
from os import listdir
from os.path import isfile, join

def get_list_of_files(path):
    list_files = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files
# def strip_file_output(elem):
    # return elem.strip('.jpg')
def create_csv_label(list_files,df_label,save_path_filename):
    # new_list_files = [strip_file_output(s) for s in list_files]
    print(df_label)

    df_label = df_label.rename(columns={'name':'filename'})
    df_label['filename'] = df_label['filename'].astype(str)+'.jpg'
    print(df_label)
    df_new = df_label[df_label['filename'].isin(list_files)]
    df_new = df_new.replace(-1,0)
    df_new.to_csv(save_path_filename,index=None)

def main():
    # Point to where the images are located (image we want to match with label csv)
    # list_img_path = ['dataset/new_dataset/positive_test','new_dataset/negative_test','new_dataset/negative_train']
    # list_img_path = ['dataset/new_dataset/positive_test','new_dataset/negative_test','new_dataset/negative_train','new_dataset/negative_img11']
    list_img_path = ['dataset/new_dataset/positive_test','dataset/new_dataset/negative_test','dataset/new_dataset/negative_train','dataset/new_dataset/negative_img11','dataset/new_dataset/negative_img12']

    list_images = []
    for path in list_img_path:
        temp_path = os.path.join(os.getcwd(), path)
        list_img = get_list_of_files(temp_path)
        list_images = list_images + list_img

    # Combine original label files
    # Note: if we have new csv file, please read csv in. As we need to combine every csv we want into one
    df_label_neg_img11 = pd.read_csv('dataset/new_dataset/negative_img_label11.csv')
    df_label_neg_img12 = pd.read_csv('dataset/new_dataset/negative_img_label12.csv')

    # df_label = pd.concat([df_label_train, df_label_test], ignore_index=True, sort =False)
    df_label = pd.concat([df_label_neg_img11,df_label_neg_img12], ignore_index=True, sort =False)

    create_csv_label(list_images,df_label,'label_test_50k.csv')

if __name__ == '__main__':
    main()
