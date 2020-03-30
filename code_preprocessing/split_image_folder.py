
''' For Image Classification '''
# positive:  8929 : 7100/1829
# negative: 52515  : 42000/10515
# train-test: 80/20 : 49100/12344
# train-> train/val: 80/20 : 39100/ 10000

''' For Object Detection (Classification+Localization)'''
# We train only on Positive dataset
# positive:  8929
# train-test: 80/20 : 7100/1829

from os import listdir
from os.path import isfile, join
from shutil import copyfile
import random
from tqdm import tqdm

NUM_POS = 8929
POS_TRAIN = 7100
POS_TEST = 1829

NUM_NEG = 52515
NEG_TRAIN = 42000
NEG_TEST = 10515

# INDEX_SPLIT = NEG_TRAIN
INDEX_SPLIT = POS_TRAIN

def get_list_of_files(path):
    list_files = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files

def move_file_to_folder(list_files,current_path,new_path):
    for i in tqdm(range(len(list_files))):
        filename = list_files[i]
        current_location = current_path+'/'+filename
        new_location = new_path+'/'+filename
        copyfile(current_location,new_location)

def split_list(a_list,num_split):
    where_split = num_split
    return a_list[:where_split], a_list[where_split:]

def main():
    type_of_file = 'positive'
    # type_of_file = 'negative'

    # Get list of files in specific path
    path = 'dataset/'+type_of_file+'_img'
    list_files = get_list_of_files(path)

    # Shuffle files in list
    random.shuffle(list_files)
    list_files_train, list_files_test = split_list(list_files,INDEX_SPLIT)

    # Move some file to folder
    new_path = 'dataset/new_dataset/'+type_of_file+'_train'
    move_file_to_folder(list_files_train,path,new_path)
    new_path = 'dataset/new_dataset/train'
    move_file_to_folder(list_files_train,path,new_path)

    new_path = 'dataset/new_dataset/'+type_of_file+'_test'
    move_file_to_folder(list_files_test,path,new_path)
    new_path = 'dataset/new_dataset/test'
    move_file_to_folder(list_files_test,path,new_path)

if __name__ =='__main__':
    main()
