import os
import numpy as np
from createFeatureDataSet import FeatureDataSet
from createTargetDataSet import TargetDataSet
from VisualizationFeature import VisualizationFeature
from model import Model
from evaluation import Evaluate
import logging
import datetime
# import Animal

class Main:

    def __init__(self):

        if os.name == 'nt':
            common_path = '..\\BSR\\BSDS500\\data\\'
        elif os.name == 'posix':
            common_path = '../BSR/BSDS500/data/'
        else:
            raise 'Unsupported OS'

        if os.name == 'nt':
            self.train_images_dir = common_path + 'images\\train'
            self.test_images_dir = common_path + 'images\\test'
            self.val_images_dir = common_path + 'images\\val'
            self.train_ground_truth_dir = common_path + 'groundTruth\\train'
            self.test_ground_truth_dir = common_path + 'groundTruth\\test'
            self.val_ground_truth_dir = common_path + 'groundTruth\\val'
        elif os.name == 'posix':
            self.train_images_dir = common_path + 'images/train'
            self.test_images_dir = common_path + 'images/test'
            self.val_images_dir = common_path + 'images/val'
            self.train_ground_truth_dir = common_path + 'groundTruth/train'
            self.test_ground_truth_dir = common_path + 'groundTruth/test'
            self.val_ground_truth_dir = common_path + 'groundTruth/val'
        else:
            raise 'Unsupported OS'

        self.visualisation_flag = False
        if os.name == 'posix':
            self.visualisation_flag = False

        self.evaluation_visualisation_flag = False      # plt.show() for F-measure
        self.abs_flag = True
        self.log_flag = True
        self.grey_picture = False           # load picture as grey/color - before feature extraction
        self.picture_resolution = [321, 481]
        self.picture_threshold = 3          # number of picture (train+test+val)
        self.label_threshold = 0.06         # min percentage of label 1 in each pictures

        self.feature_method_list = ['canny','laplacian'] # ['sobelx', 'sobely', 'canny', 'laplacian', 'scharrx', 'scharry']
        self.row_threshold = 8
        self.column_threshold = 8

        self.auto_canny = True
        self.auto_canny_sigma = 0.33

        self.learners = 'OneSlackSSVM'   # 'StructuredPerceptron'/'OneSlackSSVM'/'SubgradientSSVM'
        self.learners_parameters = {
            'max_iter' : 2,
            'verbose': 1,
            'show_loss_every' :1,
            'n_jobs': -1, # -1,                     # all cpu's
        }

        self.models = 'EdgeFeatureGraphCRF' # 'EdgeFeatureGraphCRF'         # 'GraphCRF'/'GridCRF'/'EdgeFeatureGraphCRF'
        self.models_parameters = {
            'inference': 'max-product',     # 'max-product' / 'ad3'
            'neighborhood': 4,              # 4/8
            'directed': False,
            'type': '',                     # cross_edge + models = EdgeFeatureGraphCRF + 4 neigh
            'class_weight' : list()         # if graph CRF need to add weights
        }

        self.split_train_test = True
        self.train_ratio = 0.8

        self.baseline_features = ['canny']

    def run(self):

        # load features data set (X)
        self.start_time = datetime.datetime.now()
        self.check_input_validation()
        self.init_logger()
        logging.info("start creating data set (features - X)")
        feature_data_set_obj = FeatureDataSet(
            self.train_images_dir,
            self.picture_threshold,
            self.feature_method_list,
            self.row_threshold,
            self.column_threshold,
            self.picture_resolution,
            self.label_threshold,
            self.train_ground_truth_dir,
            self.abs_flag,
            self.log_flag,
            self.grey_picture,
            self.auto_canny,
            self.auto_canny_sigma
        )
        feature_data_set_obj.run()
        logging.info("finish creating data set (features - X)")

        # load target data set (T)
        logging.info("start creating data set (target - Y)")
        target_data_set_obj = TargetDataSet(
            self.train_ground_truth_dir,
            feature_data_set_obj.images_name_list,
            self.row_threshold,
            self.column_threshold,
            feature_data_set_obj.X,
            feature_data_set_obj.pixels_frame)
        target_data_set_obj.run()
        logging.info("finish creating data set (target - Y)")

        # visualisation features+target_features
        logging.info("start visualization X+y feature")
        visualization_feature_obj = VisualizationFeature(
            feature_data_set_obj.X_canny_final,
            feature_data_set_obj.X_sobelx_final,
            feature_data_set_obj.X_sobely_final,
            feature_data_set_obj.X_laplacian_final,
            feature_data_set_obj.X_scharrx_final,
            feature_data_set_obj.X_scharry_final,
            target_data_set_obj.y,
            self.feature_method_list,
            self.dir_visualization_name,
            feature_data_set_obj.images_name_list,
            self.visualisation_flag
        )
        visualization_feature_obj.run()
        logging.info("finish visualization X+y feature")


        # run model + inference
        logging.info("start running model + inference")
        model_obj = Model(
            target_data_set_obj.X,
            target_data_set_obj.y,
            self.learners,
            self.learners_parameters,
            self.models,
            self.models_parameters,
            target_data_set_obj.total_label_one_average,
            self.row_threshold
        )
        model_obj.run()
        logging.info("finish running model + inference")

        # evaluate results
        logging.info("start evaluate results")
        evaluate_obj = Evaluate(
            model_obj.crf,
            model_obj.w,
            feature_data_set_obj.images_name_list,
            model_obj.X,
            self.train_ground_truth_dir,
            feature_data_set_obj.pixels_frame,
            self.evaluation_visualisation_flag,
            self.dir_visualization_name,
            self.models,
            feature_data_set_obj,
            self.baseline_features
        )
        evaluate_obj.run()

        self.finish_time = datetime.datetime.now()
        try:
            diff_time = self.finish_time - self.start_time
            logging.info('Diff time: ' + str(diff_time))
            div_tuple = divmod(diff_time.days * 86400 + diff_time.seconds, 60)
            logging.info('Diff time: minutes' + str(div_tuple[0]) + ', seconds: ' + str(div_tuple[1]))
        except:
            pass


        logging.info("finish evaluate results")

    # define logger properties and create file
    def init_logger(self):
        from time import gmtime, strftime
        import time
        time_in_format = strftime("%Y-%m-%d-%H-%M-%S",  time.localtime())

        self.dir_visualization_name = time_in_format + '_pic_' + str(self.picture_threshold) \
        + '_iter_' + str(self.learners_parameters['max_iter']) \
        + '_size_' + str(self.row_threshold) \
        + '_crf_' + str(self.models) \
        + '_learner_' + str(self.learners) \
        + '_inference_' + str(self.models_parameters['inference'])\
        + '_neigh_' + str(self.models_parameters['neighborhood'])


        if os.name == 'nt':
            log_dir = '..\\log\\'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_path = '..\\log\\' + self.dir_visualization_name + '.log'
        elif os.name == 'posix':
            log_dir = '../log/'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_path = '../log/' + self.dir_visualization_name + '.log'
        else:
            raise 'Unsupported OS'

        logging.basicConfig(filename=log_path,
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Start Main")
        an = Main()
        attrs = vars(an)
        logging.info('Global variables::')
        for f_n, val in attrs.iteritems():
            logging.info(str(f_n) + ': ' + str(val))
        return

    def check_input_validation(self):
        if self.column_threshold != self.row_threshold:
            raise('column and row threshold must be equivalent')
        if self.column_threshold%2 != 0:
            raise ('column and row threshold must be even (%2=0)')
        return

def main():
    main_obj = Main()
    main_obj.run()


if __name__ == "__main__":
    main()
