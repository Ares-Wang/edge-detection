import numpy as np
from arguments import Arguments
from sklearn.metrics import zero_one_loss
# from pystruct.plot_learning import plot_learning
from sklearn.metrics import f1_score
from scipy import misc
from PIL import Image
import os, os.path
import logging
import inspect

class Model:

    def __init__(self, X, y, learners, learners_parameters, models, models_parameters, total_label_one_avg, row_threshold):

        saved_args = locals()
        del saved_args['X']     # preventing from logging
        del saved_args['y']     # preventing from logging
        Arguments.logArguments('Model', saved_args)

        self.X = X
        self.y = y
        self.learners = learners
        self.learners_parameters = learners_parameters
        self.models = models
        self.models_parameters = models_parameters
        self.total_label_one_avg = total_label_one_avg
        self.row_threshold = row_threshold              # needed to define graph model

        self.total_accuracy_list = list()               # each image and relevance accuracy
        return


    def run(self):
        self.check_input()
        self.define_model()
        self.define_learners()
        self.fit_model()
        # self.inference()
        # self.train_accuracy()
        return

    def check_input(self):
        if self.learners not in ['OneSlackSSVM', 'SubgradientSSVM', 'StructuredPerceptron']:
            raise('learners is not supported: ' + str(self.learners))
        if self.models not in ['GraphCRF', 'GridCRF', 'EdgeFeatureGraphCRF']:
            raise ('learners is not supported: ' + str(self.learners))
        if self.models_parameters['neighborhood'] not in [4,8]:
            raise ('neighborhood number must be 4/8')
        return

    def define_model(self):
        if self.models == 'GridCRF':
            from pystruct.models import GridCRF
            self.crf = GridCRF(
                neighborhood=self.models_parameters['neighborhood'],
                inference_method=self.models_parameters['inference']
            )
        if self.models == 'GraphCRF':
            self.prepare_data_to_graph_crf()
            logging.info('Class weight: ' + str([self.total_label_one_avg,1-self.total_label_one_avg]))
            from pystruct.models import GraphCRF
            self.crf = GraphCRF(
                inference_method=self.models_parameters['inference'],
                directed =  self.models_parameters['directed'],
                class_weight =  [self.total_label_one_avg,1-self.total_label_one_avg]
                # class_weight=[0.01, 0.99]
            )
        if self.models == 'EdgeFeatureGraphCRF':
            self.prepare_data_to_edge_feature_graph_crf()
            logging.info('Class weight: ' + str([self.total_label_one_avg, 1 - self.total_label_one_avg]))
            from pystruct.models import EdgeFeatureGraphCRF
            self.crf = EdgeFeatureGraphCRF(
                inference_method=self.models_parameters['inference'],
                # directed=self.models_parameters['directed'],
                class_weight=[self.total_label_one_avg, 1 - self.total_label_one_avg]
                # class_weight=[0.01, 0.99]
            )
        return

    def define_learners(self):
        if self.learners == 'OneSlackSSVM':
            import pystruct.learners as ssvm
            self.clf = ssvm.OneSlackSSVM(
                model=self.crf,
                # C=100,
                # inference_cache=100,
                # tol=.1,
                verbose=self.learners_parameters['verbose'],
                max_iter=self.learners_parameters['max_iter'],
                n_jobs=self.learners_parameters['n_jobs']
            )
        if self.learners == 'SubgradientSSVM':
            import pystruct.learners as ssvm
            self.clf = ssvm.OneSlackSSVM(
                model=self.crf,
                # C=100,
                # inference_cache=100,
                # tol=.1,
                verbose=self.learners_parameters['verbose'],
                max_iter=self.learners_parameters['max_iter'],
                n_jobs=self.learners_parameters['n_jobs'],
                show_loss_every = self.learners_parameters['show_loss_every']
            )
        if self.learners == 'StructuredPerceptron':
            import pystruct.learners as structured_perceptron
            self.clf = structured_perceptron.StructuredPerceptron(
                model=self.crf,
                # C=100,
                # inference_cache=100,
                # tol=.1,
                verbose=self.learners_parameters['verbose'],
                max_iter=self.learners_parameters['max_iter'],
                n_jobs=self.learners_parameters['n_jobs'],
                # show_loss_every=self.learners_parameters['show_loss_every']
            )
        return

    def fit_model(self):
        logging.info("Start fit model")

        self.clf.fit(self.X, self.y)
        self.w = self.clf.w
        # self.objective_curve_ = self.clf.objective_curve_
        # self.training_timestamps = self.clf.timestamps_
        # self.loss_curve_ =  self.clf.loss_curve_

        logging.info('weight after fit:')
        logging.info(self.w)
        # logging.info('objective curve: (cutting plane objective)')
        # logging.info(self.objective_curve_)
        # logging.info('loss curve: (current loss)')
        # logging.info(self.loss_curve_)
        logging.info('training time each iteration:')
        # logging.info(self.training_timestamps)
        # logging.info("Finish fit model")
        return

    # not in use (in eval class we use inference)
    def inference(self):
        logging.info("Start inference - crf.inference")
        for index, cur_nd_array in enumerate(self.X):
            predict_picture = self.crf.inference(cur_nd_array, self.w)
            logging.info('inference result number: ' + str(index))
            # logging.info(predict_picture)
            # logging.info(self.y[index])
            accuracy = self.calculate_metrics(predict_picture, self.y[index])
            self.total_accuracy_list.append(accuracy)
            logging.info('accuracy: ' + str(accuracy))
            # logging.info('F1 macro: ' + str(f1_score(predict_picture, self.y[index], average='macro')))
            # logging.info('F1 micro: ' + str(f1_score(predict_picture, self.y[index], average='micro')))
        logging.info('Total accuracy: ' + str(sum(self.total_accuracy_list)/len(self.total_accuracy_list)))
        logging.info("Finish inference")
        return

    def calculate_metrics(self, predict_picture, real_picture):
        row_loss = list()
        for i, c_list in enumerate(predict_picture):
            row_loss.append(zero_one_loss(c_list, real_picture[i]))
        return 1- sum(row_loss)/len(row_loss)

    def prepare_data_to_graph_crf(self):
        from pystruct.utils import make_grid_edges, edge_list_to_features
        self.X_flatten = []
        self.y_flatten = []

        for pic_i, pic_nd_array in enumerate(self.X):
            pic_item = list()
            cell_index_place = 0
            for row_i, row_val in enumerate(pic_nd_array):      # pic item

                for col_i, cell_features in enumerate(row_val):  # pic row iteration cell by cell
                    pic_item.append(cell_features)

            if self.models_parameters['neighborhood'] == 4:
                right, down = make_grid_edges(pic_nd_array, neighborhood=4, return_lists=True)
            # right, down, upright, downright = make_grid_edges(pic_nd_array, neighborhood=8, return_lists=True)
                edges = np.vstack([right, down])
            elif self.models_parameters['neighborhood'] == 8:
                right, down, upright, downright = make_grid_edges(pic_nd_array, neighborhood=8, return_lists=True)
                edges = np.vstack([right, down, upright, downright])
            # for val in range

            # Guy version - old
            # edges_item = list()
            # max_cell_index = self.row_threshold * self.row_threshold
            # last_row_first_index = max_cell_index - self.row_threshold      # e.g. 36-6
            # for i in range(0, max_cell_index):
            #     if i<last_row_first_index:              # except last row
            #         edges_item.append(np.array([i, i + self.row_threshold]))
            #
            #     if (i+1)%self.row_threshold != 0:       # except last col
            #         edges_item.append(np.array([i, i + 1]))

            # finish iterate picture
            self.X_flatten.append((np.array(pic_item), edges))

        for pic_i, pic_nd_array in enumerate(self.y):
            pic_item = list()
            for row_i, row_val in enumerate(pic_nd_array):  # pic item

                for col_i, cell_features in enumerate(row_val):  # pic row iteration cell by cell
                    pic_item.append(cell_features)

            self.y_flatten.append(pic_item)

        self.X = np.array(self.X_flatten)
        self.y = np.array(self.y_flatten)
        return

    def prepare_data_to_edge_feature_graph_crf(self):
        from pystruct.utils import make_grid_edges, edge_list_to_features
        self.X_flatten = []
        self.y_flatten = []

        for pic_i, pic_nd_array in enumerate(self.X):
            pic_item = list()
            for row_i, row_val in enumerate(pic_nd_array):          # pic item

                for col_i, cell_features in enumerate(row_val):     # pic row iteration cell by cell
                    pic_item.append(cell_features)

            if self.models_parameters['neighborhood'] == 4:

                # 4 neigh and cross type
                if 'type' in self.models_parameters and self.models_parameters['type'] == 'cross_edge':
                    upright, downright = self.cross_make_grid_edges(pic_nd_array, neighborhood=4, return_lists=True)
                    edges = np.vstack([upright, downright])
                    edge_features_directions = self.edge_list_to_features([upright, downright], 4)
                else:   # regular
                    right, down = make_grid_edges(pic_nd_array, neighborhood=4, return_lists=True)
                    edges = np.vstack([right, down])
                    edge_features_directions = self.edge_list_to_features([right, down], 4)

            elif self.models_parameters['neighborhood'] == 8:
                right, down, upright, downright = make_grid_edges(pic_nd_array, neighborhood=8, return_lists=True)
                edges = np.vstack([right, down, upright, downright])
                edge_features_directions = self.edge_list_to_features([right, down, upright, downright], 8)
            # finish iterate picture - (pixel feature (list), edge (pixel-pixel) list)
            self.X_flatten.append((np.array(pic_item), edges, edge_features_directions))

        for pic_i, pic_nd_array in enumerate(self.y):
            pic_item = list()
            for row_i, row_val in enumerate(pic_nd_array):          # pic item

                for col_i, cell_features in enumerate(row_val):     # pic row iteration cell by cell
                    pic_item.append(cell_features)

            self.y_flatten.append(pic_item)

        self.X = np.array(self.X_flatten)
        self.y = np.array(self.y_flatten)
        return

    # add direction feature to each edge
    def edge_list_to_features(self, edge_list, neighborhood):
        edges = np.vstack(edge_list)

        if neighborhood == 4:
            edge_features = np.zeros((edges.shape[0], 2))
            edge_features[:len(edge_list[0]), 0] = 1
            edge_features[len(edge_list[0]):, 1] = 1

        elif neighborhood == 8:

            limit_0 = len(edge_list[0])
            limit_1 = len(edge_list[0]) + len(edge_list[1])
            limit_2 = len(edge_list[0]) + len(edge_list[1]) + len(edge_list[2])
            limit_3 = len(edge_list[0]) + len(edge_list[1]) + len(edge_list[2]) + + len(edge_list[3])

            edge_features = np.zeros((edges.shape[0], 4))
            edge_features[:limit_0, 0] = 1
            edge_features[limit_0:limit_1, 1] = 1
            edge_features[limit_1:limit_2, 2] = 1
            edge_features[limit_2:limit_3, 3] = 1
        else:
            raise('number neighborhood is not supported 4/8')
        return edge_features

    def cross_make_grid_edges(self, x, neighborhood=4, return_lists=False):
        if neighborhood not in [4]:
            raise ValueError("neighborhood can only be '4', got %s" %
                             repr(neighborhood))
        inds = np.arange(x.shape[0] * x.shape[1]).reshape(x.shape[:2])
        inds = inds.astype(np.int64)

        upright = np.c_[inds[1:, :-1].ravel(), inds[:-1, 1:].ravel()]
        downright = np.c_[inds[:-1, :-1].ravel(), inds[1:, 1:].ravel()]
        edges = [upright, downright]

        if return_lists:
            return edges
        return np.vstack(edges)

def main():
    print('start create model')
    main_obj = Model()
    main_obj.run()
    print('finish create model')

if __name__ == "__main__":
    main()
