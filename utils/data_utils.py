import os
import pickle

yuluo_file_names = ['../data/Conus-betulinus/betulinus-conotoxin.txt', '../data/Conus-betulinus/betulinus-non-conotoxin.txt',
                    '../data/Lautoconus-ventricosus/Lautoconus-ventricosus-conotoxin.txt', '../data/Lautoconus-ventricosus/Lautoconus-non-conotoxin.txt']

path_to_save = ['../data/yuluo.dat', '../data/non-yuluo.dat']

def ParseData(data_path):

    if '-non-' in data_path:
        tag = 0
    else:
        tag = 1

    with open(data_path) as infile:
        lines = infile.readlines()

    data = []
    for line in lines:
        if lines == '>':
            continue
        else:
            sample = {'seq': line, 'tag': tag}
        data.append(sample)
    return data

def read_data_from_file(file_names):
    """
    :param file_names: ames of files to process
    :return: two lists: pos_data, neg_data
    """
    pos_data = []
    neg_data = []
    for file_name in file_names:
        data = ParseData(file_name)
        if '-non-' in file_name:
            neg_data.extend(data)
        else:
            pos_data.extend(data)
    return pos_data, neg_data

def ProcessData():
    pos_data, neg_data = read_data_from_file(file_names=yuluo_file_names)
    pickle.dump(pos_data, open(path_to_save[0], 'wb'))
    pickle.dump(neg_data, open(path_to_save[1], 'wb'))

ProcessData()