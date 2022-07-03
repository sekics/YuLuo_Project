import os
import pickle

yuluo_file_names = ['../data/Conus-betulinus/betulinus-conotoxin.txt', '../data/Conus-betulinus/betulinus-non-conotoxin.txt',
                    '../data/Lautoconus-ventricosus/Lautoconus-ventricosus-conotoxin.txt', '../data/Lautoconus-ventricosus/Lautoconus-non-conotoxin.txt']

path_to_save = ['yuluo.bin', 'non-yuluo.bin']
def read_data_from_file(file_names):
    """
    :param file_names: ames of files to process
    :return: two lists: pos_data, neg_data
    """
    pos_data = []
    neg_data = []
    for file in file_names:

        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            if "non" in file:
                if line[0] == '>':
                    pass
                else:
                    sample = {'seq': line, 'tag': 0}
                neg_data.append(sample)
            else:
                if line[0] == '>':
                    pass
                else:
                    sample = {'seq': line, 'tag': 1}
                pos_data.append(sample)

    return pos_data, neg_data

def ProcessData():
    pos_data, neg_data = read_data_from_file(file_names=yuluo_file_names)
    pickle.dump(pos_data, path_to_save[0])
    pickle.dump(neg_data, path_to_save[1])
