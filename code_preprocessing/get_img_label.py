from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm

'''Getting image labels from original dataset (which contains all label train/test csvfile in folder (10,100,1000)) '''


def get_list_of_files(path):
    list_files = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files
def strip_file_output(elem):
    return elem.strip('.jpg')
def create_csv_label(list_files,df_label,save_path_filename):
    new_list_files = [strip_file_output(s) for s in list_files]

    df_new = df_label[df_label['name'].isin(new_list_files)]
    print(df_new)
    df_new.to_csv(save_path_filename,index=None)

WHICH_FILE = 'test'
# WHICH_FILE = 'train'

def main():
    ''' ImageSet contains the filename and label classes of all data '''
    list_folder = ['10','100','1000']
    path_label = 'dataset/ImageSet'

    # Can change path (path:image path, save_path_filename: where to save csv)
    # path = 'dataset/new_dataset/'+WHICH_FILE
    path = 'dataset/new_dataset/negative_img12'

    # save_path_filename = 'dataset/new_dataset/label/'+WHICH_FILE+'.csv'
    save_path_filename = 'dataset/new_dataset/negative_img_label12.csv'

    # Get list of files in specific path
    list_files = get_list_of_files(path)


    index = 0
    for label_folder in list_folder:
        path_to_label = path_label+'/'+label_folder
        for file in tqdm(listdir(path_to_label)):
            fullpath_to_label = path_to_label+'/'+file
            temp_df = pd.read_csv(fullpath_to_label)
            if index ==0:
                df = temp_df.copy()
                index += 1
                continue
            df = pd.concat([df,temp_df],axis=0)
            print(file)
    final_df = df.drop_duplicates(subset=['name'], keep='first')
    create_csv_label(list_files,final_df,save_path_filename)

if __name__ =='__main__':
    main()
