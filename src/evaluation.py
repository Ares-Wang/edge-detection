from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from arguments import Arguments
import scipy.io
import os, os.path
import logging
import operator
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class Evaluate:

    def __init__(self, crf, w, image_name_list, X, ground_truth_dir, pixel_frame, evaluation_visualisation_flag,
                 dir_visualization_name, model_name, feature_data_set_obj, baseline_features=None):

        saved_args = locals()
        del saved_args['X']  # preventing from logging
        Arguments.logArguments('Evaluate', saved_args)

        self.crf = crf
        self.w = w
        self.X = X
        self.image_name_list = image_name_list
        self.ground_truth_dir = ground_truth_dir
        self.pixels_frame = pixel_frame
        self.evaluation_visualisation_flag = evaluation_visualisation_flag
        self.dir_visualization_name = dir_visualization_name
        self.model_name = model_name

        self.baseline_features_dict = dict()
        if len(baseline_features) > 0:
            for f in baseline_features:
                attr = getattr(feature_data_set_obj, 'X_{}_final'.format(f))
                self.baseline_features_dict[f] = attr

        self.all_f1_avg = list()        # avg f1 (between all gt)
        self.all_f1 = list()            # max f1 (max per gt)
        self.all_recall = list()
        self.all_precision = list()

        self.all_super_f1_avg = list()
        self.all_super_f1 = list()
        self.all_super_recall = list()
        self.all_super_precision = list()

        self.canny_f1_avg = list()
        self.canny_f1 = list()
        self.canny_recall = list()
        self.canny_precision = list()

        self.canny_super_f1_avg = list()
        self.canny_super_f1 = list()
        self.canny_super_recall = list()
        self.canny_super_precision = list()
        return

    def run(self):
        self.create_dir()
        self.find_all_edge_connetion()      # calculate all edge connection for super pixel once
        self.evaluate_max_f_measure()
        self.evaluate_max_f_measure_canny()
        self.final_result()                 # after examine all pictures - show summary results
        self.final_result_canny()
        return

    def create_dir(self):
        if os.name == 'nt':
            vis_dir = '..\\visualisation_eval\\'
        elif os.name == 'posix':
            vis_dir = '../visualisation_eval/'

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if os.name == 'nt':
            vis_in_dir = '..\\visualisation_eval\\' + str(self.dir_visualization_name) + '\\'
        elif os.name == 'posix':
            vis_in_dir = '../visualisation_eval/' + str(self.dir_visualization_name) + '/'

        if not os.path.exists(vis_in_dir):
            os.makedirs(vis_in_dir)

        self.vis_in_dir = vis_in_dir
        return

    # for each file find is maximum f measure
    def evaluate_max_f_measure(self):
        logging.info('start calculating algorithm performance...')
        self.row_range = range(self.pixels_frame[0], self.pixels_frame[1])
        self.column_range = range(self.pixels_frame[2], self.pixels_frame[3])

        for index, image_name in enumerate(self.image_name_list):
            logging.info('')
            logging.info('calculate max f-measure picture number: ' + str(index))
            pic_path = os.path.join(self.ground_truth_dir, image_name)
            mat = scipy.io.loadmat(pic_path)

            # inference once of picture
            predict_picture = self.crf.inference(self.X[index], self.w)

            # iterate over all ground truth pictures
            pic_metrics = list()        # store all ground truth metrics
            pic_f1 = list()
            pic_recall = list()
            pic_precision = list()
            pic_super_f1 = list()
            pic_super_recall = list()
            pic_super_precision = list()

            # iterate all ground true
            for gt_index, pic_matrix in enumerate(mat['groundTruth'][0]):
                pic_after_cut = list()
                all_picture = mat['groundTruth'][0][gt_index]['Boundaries'][0][0]
                pic_after_cut = all_picture[self.row_range, :][:, self.column_range]

                # if mode graphCRF flatten ground truth to compare
                if self.model_name == 'GraphCRF':
                    pic_after_cut = np.array(pic_after_cut).flatten()
                cur_metrics = self.calculate_f_measure(predict_picture, pic_after_cut)

                pic_metrics.append(cur_metrics)
                pic_f1.append(cur_metrics['f1'])
                pic_recall.append(cur_metrics['recall'])
                pic_precision.append(cur_metrics['precision'])

                super_pixel_res = self.calculate_super_pixel_measure(predict_picture, pic_after_cut)
                pic_super_f1.append(super_pixel_res['f_measure'])
                pic_super_recall.append(super_pixel_res['recall'])
                pic_super_precision.append(super_pixel_res['precision'])

            logging.info('max f1 (between all gt): ' + str(max(pic_f1)))
            logging.info('avg f1 (between all gt): ' + str(sum(pic_f1)/len(pic_f1)))
            logging.info('max super f1 (between all gt): ' + str(max(pic_super_f1)))
            logging.info('avg super f1 (between all gt): ' + str(sum(pic_super_f1) / len(pic_super_f1)))

            self.all_f1_avg.append(sum(pic_f1)/len(pic_f1)) # avg of all ground true per image
            self.all_f1.append(max(pic_f1))                 # max of all ground true per image
            self.all_recall.append(max(pic_recall))
            self.all_precision.append(max(pic_precision))

            self.all_super_f1_avg.append(sum(pic_super_f1) / len(pic_super_f1)) # avg of all ground true per image
            self.all_super_f1.append(max(pic_super_f1))                         # max of all ground true per image
            self.all_super_recall.append(max(pic_super_recall))
            self.all_super_precision.append(max(pic_super_precision))

            # if self.evaluation_visualisation_flag:        # print all g_t and predict pictures
            self.visualisation_eval(mat, predict_picture, image_name, pic_f1, max(pic_f1), pic_super_f1, max(pic_super_f1), 'CRF')

        logging.info('finish calculating algorithm performance...')
        return
    
    def evaluate_max_f_measure_canny(self):

        logging.info('')
        logging.info('start calculating Canny performance...')
        self.row_range = range(self.pixels_frame[0], self.pixels_frame[1])
        self.column_range = range(self.pixels_frame[2], self.pixels_frame[3])

        for index, image_name in enumerate(self.image_name_list):
            logging.info('calculate baseline max f-measure picture number: ' + str(index))
            pic_path = os.path.join(self.ground_truth_dir, image_name)
            mat = scipy.io.loadmat(pic_path)

            # iterate over all ground truth pictures
            predict_picture = self.baseline_features_dict['canny'][index]

            # no need to normalize as you already did in createFeatureDataSet
            # predict_picture[predict_picture == 255] = 1

            pic_metrics = list()        # store all ground truth metrics
            pic_f1 = list()
            pic_recall = list()
            pic_precision = list()
            pic_super_f1 = list()
            pic_super_recall = list()
            pic_super_precision = list()
            for gt_index, pic_matrix in enumerate(mat['groundTruth'][0]):
                pic_after_cut = list()
                all_picture = mat['groundTruth'][0][gt_index]['Boundaries'][0][0]
                pic_after_cut = all_picture[self.row_range, :][:, self.column_range]
                cur_metrics = self.calculate_f_measure(predict_picture, pic_after_cut)

                pic_metrics.append(cur_metrics)
                pic_f1.append(cur_metrics['f1'])
                pic_recall.append(cur_metrics['recall'])
                pic_precision.append(cur_metrics['precision'])

                super_pixel_res = self.calculate_super_pixel_measure(predict_picture, pic_after_cut)
                pic_super_f1.append(super_pixel_res['f_measure'])
                pic_super_recall.append(super_pixel_res['recall'])
                pic_super_precision.append(super_pixel_res['precision'])

            logging.info('max f1 (between all gt): ' + str(max(pic_f1)))
            logging.info('avg f1 (between all gt): ' + str(sum(pic_f1) / len(pic_f1)))
            logging.info('max super f1 (between all gt): ' + str(max(pic_super_f1)))
            logging.info('avg super f1 (between all gt): ' + str(sum(pic_super_f1) / len(pic_super_f1)))

            self.canny_f1_avg.append(sum(pic_f1) / len(pic_f1))     # avg of all ground true per image
            self.canny_f1.append(max(pic_f1))
            self.canny_recall.append(max(pic_recall))
            self.canny_precision.append(max(pic_precision))

            self.canny_super_f1_avg.append(sum(pic_super_f1) / len(pic_super_f1))  # avg of all ground true per image
            self.canny_super_f1.append(max(pic_super_f1))  # max of all ground true per image
            self.canny_super_recall.append(max(pic_super_recall))
            self.canny_super_precision.append(max(pic_super_precision))

            # if self.evaluation_visualisation_flag:                # print all g_t and predict pictures
            self.visualisation_eval(mat, predict_picture, image_name, pic_f1, max(pic_f1), pic_super_f1, max(pic_super_f1), 'Canny')

        logging.info('finish calculating Canny performance...')
        return

    # plot our prediction and first 5 ground true
    def visualisation_eval(self, mat, predict_picture, image_name, pic_f1, max_pic_f1, pic_super_f1,
                           max_pic_super_f1, folder_name):

        if os.name == 'nt':
            vis_dir = self.vis_in_dir + folder_name + '\\'
        elif os.name == 'posix':
            vis_dir = self.vis_in_dir + folder_name + '/'

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # un flatten (reshape) predict image to expected shape (row threshold, column threshold)
        if self.model_name in ['GraphCRF', 'EdgeFeatureGraphCRF']:
            row_dim = self.pixels_frame[1] - self.pixels_frame[0]
            col_dim = self.pixels_frame[3] - self.pixels_frame[2]
            predict_picture = predict_picture.reshape((row_dim, col_dim))

        fig = plt.figure()
        a = fig.add_subplot(2, 3, 1)
        a.set_title('predict')
        plt.imshow(predict_picture)
        for gt_index, pic_matrix in enumerate(mat['groundTruth'][0]):

            pic_after_cut = list()
            all_picture = mat['groundTruth'][0][gt_index]['Boundaries'][0][0]
            pic_after_cut = all_picture[self.row_range, :][:, self.column_range]

            if gt_index + 2 <= 6:
                a = fig.add_subplot(2, 3, gt_index+2)
                a.set_title('gt #' + str(gt_index) + ', F1: ' + str(round(pic_f1[gt_index],3)) + ', SF1: ' + str(round(pic_super_f1[gt_index],3)))
                plt.imshow(pic_after_cut)


        plt.savefig(vis_dir + str(image_name) + '-F1-'+ str(round(max_pic_f1,3)) + '-SF1-' + str(round(max_pic_super_f1,3)) + '.png', bbox_inches='tight')
        if os.name == 'nt' and self.evaluation_visualisation_flag:
            plt.show()
        plt.close()

        return

    # calculate metrics (f-measure, recall, precision) for one picture
    def calculate_f_measure(self, predict_picture, ground_truth_picture):
        predict_picture_flatten = predict_picture.flatten()
        ground_truth_picture_flatten = ground_truth_picture.flatten()

        f1_score_avg = f1_score(ground_truth_picture_flatten, predict_picture_flatten)
        precision_score_avg = precision_score(ground_truth_picture_flatten, predict_picture_flatten)
        recall_score_avg = recall_score(ground_truth_picture_flatten, predict_picture_flatten)

        # for predict_row in predict_picture:
        #     truth_row = ground_truth_picture[row_i]
        #     precision_macro = precision_score(truth_row, predict_row, labels=[0,1], average='macro')
        #     precision_micro = precision_score(truth_row, predict_row, labels=[0,1], average='micro')
        #     precision_binary = precision_score(truth_row, predict_row, labels=[0,1], average='binary')
        #
        #     recall_score_macro = recall_score(truth_row, predict_row, average='macro')
        #     recall_score_micro = recall_score(truth_row, predict_row, average='micro')
        #     recall_score_binary = recall_score(truth_row, predict_row, average='binary')
        #
        #     f1_score_macro = f1_score(truth_row, predict_row, average='macro')
        #     f1_score_micro = f1_score(truth_row, predict_row, average='micro')
        #     f1_score_binary = f1_score(truth_row, predict_row, average='binary')
        #
        #     f1_score_list.append(f1_score_micro)
        #     precision_score_list.append(precision_micro)
        #     recall_score_list.append(recall_score_micro)
        #     row_i +=1
        #
        # f1_score_avg = sum(f1_score_list) / len(f1_score_list)
        # precision_score_avg = sum(precision_score_list) / len(precision_score_list)
        # recall_score_avg = sum(recall_score_list) / len(recall_score_lis
        return {
            'f1': f1_score_avg,
            'precision': precision_score_avg,
            'recall': recall_score_avg
        }

    def calculate_super_pixel_measure(self, predict_picture, ground_truth_picture):

        predict_picture_flatten = predict_picture.flatten()
        ground_truth_picture_flatten = ground_truth_picture.flatten()
        ground_truth_picture_flatten= np.array(ground_truth_picture_flatten)

        TP = 0  # True Positive
        TN = 0  # True Negative
        FP = 0  # False Positive
        FN = 0  # False Negative

        for pixel_idx, surround_pixels_list in self.pixel_dict.iteritems():

            if predict_picture_flatten[pixel_idx] == 1:
                surround_pixels_list.append(pixel_idx)
                all_neigh_label = list(ground_truth_picture_flatten[surround_pixels_list])

                if 1 in all_neigh_label:
                    TP += 1
                else:
                    FP += 1

            if predict_picture_flatten[pixel_idx] == 0:
                if ground_truth_picture_flatten[pixel_idx] == 0:
                    TN += 1
                else:
                    FN += 1

        if FN+TP != 0:
            recall = float(TP)/float(FN+TP)
        else:
            recall = 0

        if TP+FP != 0:
            precision = float(TP)/float(TP+FP)
        else:
            precision = 0

        if precision+recall != 0:
            f_measure = float(2*(precision*recall))/float(precision+recall)
        else:
            f_measure = 0

        # print(ground_truth_picture_flatten)
        # print({
        #     'recall' : recall,
        #     'precision': precision,
        #     'f_measure': f_measure
        # })
        return {
            'recall': recall,
            'precision': precision,
            'f_measure': f_measure
        }


    # calculate for each pixel his all surround pixels
    def find_all_edge_connetion(self):
        row_dim = self.pixels_frame[1] - self.pixels_frame[0]
        col_dim = self.pixels_frame[3] - self.pixels_frame[2]
        index_2_dim_array = np.zeros((row_dim, col_dim), dtype='int32')
        from pystruct.utils import make_grid_edges, edge_list_to_features
        right, down, upright, downright = make_grid_edges(index_2_dim_array, neighborhood=8, return_lists=True)
        edges = np.vstack([right, down, upright, downright])
        total_num_cell = row_dim*col_dim

        pixel_dict = dict()

        for c_num in range(0, total_num_cell):
            pixel_dict[c_num] = list()

            list_tuple_indexes = zip(*np.where(edges == c_num)) # find pixel idx in all edges 2d-array
            for i, c_tuple in enumerate(list_tuple_indexes):
                c_edge_index = c_tuple[0]
                c_edge_place = c_tuple[1]
                if c_edge_place == 0:                           # find surround pixel per edges
                    pixel_dict[c_num].append(edges[c_edge_index][1])
                elif c_edge_place == 1:
                    pixel_dict[c_num].append(edges[c_edge_index][0])

        self.pixel_dict = pixel_dict
        return

    # log aggregated results
    def final_result(self):
        logging.info('')
        logging.info('Algorithm final_result')
        # logging.info('F1: ' + str(self.all_f1))
        # logging.info('Recall: ' + str(self.all_recall))
        # logging.info('Precision: ' + str(self.all_precision))

        self.avg_f1 = sum(self.all_f1_avg) / len(self.all_f1_avg)
        self.max_f1 = sum(self.all_f1) / len(self.all_f1)
        self.max_recall = sum(self.all_recall) / len(self.all_recall)
        self.max_precision = sum(self.all_precision) / len(self.all_precision)

        logging.info('')
        logging.info('Avg F1 average: ' + str(self.avg_f1))
        logging.info('Max F1 average: ' + str(self.max_f1))
        logging.info('Max Recall average: ' + str(self.max_recall))
        logging.info('Max Precision average: ' + str(self.max_precision))

        self.avg_super_f1 = sum(self.all_super_f1_avg) / len(self.all_super_f1_avg)
        self.max_super_f1 = sum(self.all_super_f1) / len(self.all_super_f1)
        self.max_super_recall = sum(self.all_super_recall) / len(self.all_super_recall)
        self.max_super_precision = sum(self.all_super_precision) / len(self.all_super_precision)

        logging.info('')
        logging.info('Avg Super F1 average: ' + str(self.avg_super_f1))
        logging.info('Max Super F1 average: ' + str(self.max_super_f1))
        logging.info('Max Super Recall average: ' + str(self.max_super_recall))
        logging.info('Max Super Precision average: ' + str(self.max_super_precision))

        # logging.info('finish algorithm final_result')
        return

    def final_result_canny(self):
        logging.info('')
        logging.info('Canny final_result')
        # logging.info('F1: ' + str(self.canny_f1))
        # logging.info('Recall: ' + str(self.canny_recall))
        # logging.info('Precision: ' + str(self.canny_precision))

        self.avg_f1_canny = sum(self.canny_f1_avg) / len(self.canny_f1_avg)
        self.max_f1_canny = sum(self.canny_f1) / len(self.canny_f1)
        self.max_recall_canny = sum(self.canny_recall) / len(self.canny_recall)
        self.max_precision_canny = sum(self.canny_precision) / len(self.canny_precision)

        logging.info('')
        logging.info('Avg F1 average: ' + str(self.avg_f1_canny))
        logging.info('Max F1 average: ' + str(self.max_f1_canny))
        logging.info('Max Recall average: ' + str(self.max_recall_canny))
        logging.info('Max Precision average: ' + str(self.max_precision_canny))

        self.avg_super_f1_canny = sum(self.canny_super_f1_avg) / len(self.canny_super_f1_avg)
        self.max_super_f1_canny = sum(self.canny_super_f1) / len(self.canny_super_f1)
        self.max_super_recall_canny = sum(self.canny_super_recall) / len(self.canny_super_recall)
        self.max_super_precision_canny = sum(self.canny_super_precision) / len(self.canny_super_precision)

        logging.info('')
        logging.info('Avg Super F1 average: ' + str(self.avg_super_f1_canny))
        logging.info('Max Super F1 average: ' + str(self.max_super_f1_canny))
        logging.info('Max Super Recall average: ' + str(self.max_super_recall_canny))
        logging.info('Max Super Precision average: ' + str(self.max_super_precision_canny))
        # logging.info('finish Canny final_result')
        return

def main():
    print('Evaluate class can not run from main')

if __name__ == "__main__":
    main()
