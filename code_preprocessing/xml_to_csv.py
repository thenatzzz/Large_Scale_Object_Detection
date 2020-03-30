import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

'''Turning xml files into csv file based on important field names '''

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print(xml_file)
        for member in root.findall('object'):
            try:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         float(member[4][0].text),
                         float(member[4][1].text),
                         float(member[4][2].text),
                         float(member[4][3].text)
                         )
            except:
                pass
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df = xml_df.drop_duplicates()
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'Annotation_positive')

    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('xray.csv', index=None)
    print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()
