import scipy.io
import os, os.path
import numpy as np
import logging
from arguments import Arguments

class TargetDataSet:

    def __init__(self, ground_truth_dir, images_name_list, row_threshold, column_threshold, X, pixels_frame):

        saved_args = locals()
        del saved_args['X']  # preventing from logging
        Arguments.logArguments('TargetDataSet', saved_args)
        self.ground_truth_dir = ground_truth_dir
        self.images_name_list = images_name_list
        self.row_threshold = row_threshold
        self.column_threshold = column_threshold
        self.pixels_frame = pixels_frame
        self.y = list()
        self.X = X

        self.delete_indexes = list()

        return

    def run(self):
        self.load_label_data()
        self.examine_statistical_train_issues()
        return

    # load label picture in shape regards to row and col threshold
    def load_label_data(self):
        for index, file_name in enumerate(self.images_name_list):
            pic_path = os.path.join(self.ground_truth_dir, file_name)
            mat = scipy.io.loadmat(pic_path)
            # row_range = range(11, 17)
            row_range = range(self.pixels_frame[0], self.pixels_frame[1])
            column_range = range(self.pixels_frame[2], self.pixels_frame[3])
            if self.check_dimension_fit(mat['groundTruth'][0][0]['Boundaries'][0][0], index):        # check shape fit
                # extract relevant shape
                pic_after_cut = list()
                all_picture = mat['groundTruth'][0][0]['Boundaries'][0][0]
                pic_after_cut = all_picture[row_range, :][:, column_range]
                '''
                cut_rows_ndarray = all_picture[:, [self.pixels_frame[0],self.pixels_frame[1]]]
                # cut_rows_ndarray = all_picture[:self.row_threshold]
                for row in cut_rows_ndarray:
                    # cut_row = row[:self.column_threshold]
                    cut_row = row[:, [self.pixels_frame[2],self.pixels_frame[3]]]
                    pic_after_cut.append(np.array(cut_row))'''
                self.y.append(np.array(pic_after_cut))

        self.delete_all_miss_match()
        return

    def check_dimension_fit(self, label_matlab_pic, source_index_pic):
        return True
        if label_matlab_pic.shape[0] != 321 or label_matlab_pic.shape[1] != 481:
            self.delete_indexes.append(source_index_pic)
            return False

        if label_matlab_pic.shape[0] == self.X[source_index_pic].shape[0] \
                and label_matlab_pic.shape[1] == self.X[source_index_pic].shape[1]:
            return True

        self.delete_indexes.append(source_index_pic)
        return False

    def delete_all_miss_match(self):
        # print(self.delete_indexes)
        tuple_index = tuple(self.delete_indexes)
        # print(len(self.X))
        self.X = [i for j, i in enumerate(self.X) if j not in tuple_index]
        # print(len(self.X))
        # print(len(self.images_name_list))
        self.images_name_list = [i for j, i in enumerate(self.images_name_list) if j not in tuple_index]
        # print(len(self.images_name_list))
        return

    def examine_statistical_train_issues(self):
        self.label_one_percentage = list()
        for y_ndarray in self.y:
            self.label_one_percentage.append(np.average(y_ndarray))

        self.total_label_one_average = sum(self.label_one_percentage)/len(self.label_one_percentage)
        logging.info('Label one minimum: ' + str(min(self.label_one_percentage)))
        logging.info('Label one maximum: ' + str(max(self.label_one_percentage)))
        logging.info('Label one average: ' + str(self.total_label_one_average))
        logging.info('Majority class: ' + str(1-self.total_label_one_average))
        logging.info('Label one all results: ' + str(self.label_one_percentage))
        return

def main():
    print('start create target data set basic model')
    main_obj = TargetDataSet()
    main_obj.run()
    print('finish LBP basic model')

if __name__ == "__main__":
    main()
