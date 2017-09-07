import numpy as np
from scipy import misc
from PIL import Image
from arguments import Arguments
import os, os.path
import logging

class FeatureDataSet:

    def __init__(self, images_dir, picture_amount_threshold, feature_method_list, row_threshold, column_threshold,
                 picture_resolution, label_threshold, ground_truth_dir, abs_flag, log_flag, grey_picture=False, auto_canny=True, auto_canny_sigma=0.33):

        saved_args = locals()
        Arguments.logArguments('FeatureDataSet', saved_args)

        self.create_grey_dir = False
        self.images_dir = images_dir
        self.picture_amount_threshold = picture_amount_threshold        # number of pictures
        self.grey_picture = grey_picture                                # load picture as grey/color
        self.feature_method_list = feature_method_list                  # list of feature extraction method
        self.row_threshold = row_threshold
        self.column_threshold = column_threshold
        self.picture_resolution = picture_resolution
        self.label_threshold = label_threshold                          # min amount of label 1 percentage
        self.ground_truth_dir = ground_truth_dir    # dir of ground true pictures - extract only high class 1 percent
        self.abs_flag = abs_flag
        self.log_flag = log_flag

        self.images_pixel_dict = dict()                                 # contain images and their pixels
        self.images_name_list = list()                                  # contain pic name in data set
        self.X = list()                                                 # list of ndarray - feature format
        self.X_sobelx = list()
        self.X_sobely = list()
        self.X_laplacian = list()
        self.X_canny = list()
        self.X_scharrx = list()
        self.X_scharry = list()
        self.X_sobelx_final = list()
        self.X_sobely_final = list()
        self.X_laplacian_final = list()
        self.X_canny_final = list()
        self.X_scharrx_final = list()
        self.X_scharry_final = list()

        self.auto_canny = auto_canny
        self.auto_canny_sigma = auto_canny_sigma
        if self.auto_canny_sigma < 0 or self.auto_canny_sigma > 1:
            raise 'Illegal value for auto Canny Sigma parameter - need to be 0<=..<=1'

        self.kepsilon = 1e-8

        return

    def run(self):
        self.check_input()
        self.define_center_pixels() # define threshold [x_min, x_max, y_min, y_max]
        self.load_pictures()        # load pictures - fit size and more
        self.extract_feature()      # sober, canny
        self.merge_feature()
        return

    def check_input(self):
        if self.grey_picture != False:
            raise('grey picture variable currently only support False bool value')
        return

    # regards to threshold size and picture size determine [x_min, x_max, y_min, y_max] of picture.
    def define_center_pixels(self):

        row_middle = self.picture_resolution[0] / 2
        row_min = row_middle - (self.row_threshold / 2)
        row_max = row_middle + (self.row_threshold / 2)

        column_middle = self.picture_resolution[1] / 2
        column_min = column_middle - (self.column_threshold / 2)
        column_max = column_middle + (self.column_threshold / 2)

        self.pixels_frame = [row_min, row_max, column_min, column_max]
        logging.debug('row min: ' + str(row_min))
        logging.debug('row max: ' + str(row_max))
        logging.debug('column min: ' + str(column_min))
        logging.debug('column max: ' + str(column_max))
        return

    def load_pictures(self):
        num_files = len(os.walk(self.images_dir).next()[2])
        count_file = 1

        for filename in os.listdir(self.images_dir):
            if count_file % 50 == 0:
                logging.debug(str(count_file) + ' / ' + str(num_files))

            if filename[-4:] != '.jpg':
                continue

            if count_file > self.picture_amount_threshold:
                break       # already load properly amount pictures

            pic_path = os.path.join(self.images_dir, filename)
            filename = filename[:-4]        # without .jpg

            # load pictures
            img_array = misc.imread(pic_path, self.grey_picture)
            if img_array.shape[0] != self.picture_resolution[0] or img_array.shape[1] != self.picture_resolution[1]:
                continue

            # check if label 1 percentage above threshold
            if self.label_percentage_valid_threshold(filename):
                # load only good pictures
                count_file += 1
                self.images_pixel_dict[filename] = img_array
                self.images_name_list.append(filename)

        return

    def extract_feature(self):
        if 'sobelx' in self.feature_method_list:
            logging.info("extract sobelx feature")
            self.add_sobelx_feature()
        if 'sobely' in self.feature_method_list:
            logging.info("extract sobely feature")
            self.add_sobely_feature()
        if 'laplacian' in self.feature_method_list:
            logging.info("extract laplacian feature")
            self.add_laplacian_feature()
        if 'canny' in self.feature_method_list:
            logging.info("extract canny feature")
            self.add_canny_feature()
        if 'scharrx' in self.feature_method_list:
            logging.info("extract scharrx feature")
            self.add_scharrx_feature()
        if 'scharry' in self.feature_method_list:
            logging.info("extract scharry feature")
            self.add_scharry_feature()

        return

    # extract relevant cell regards threshold and features
    def merge_feature(self):
        for idx, pic_ndarray in enumerate(self.X_canny):

            logging.info(('merge features - picture number: ' + str(idx)))

            cur_pic_final_format = list()
            cur_pic_canny_final_format = list()
            cur_pic_sobelx_final_format = list()
            cur_pic_sobely_final_format = list()
            cur_pic_laplacian_final_format = list()
            cur_pic_scharrx_final_format = list()
            cur_pic_scharry_final_format = list()

            row_i = 0
            for row in pic_ndarray:
                # if row_i >= self.row_threshold:
                #     continue
                # print(row_i)
                if row_i < self.pixels_frame[0] or row_i >= self.pixels_frame[1]:
                    row_i += 1
                    continue
                cur_pic_final_row = list()
                canny_row = list()
                sobelx_row = list()
                sobely_row = list()
                laplacian_row = list()
                scharrx_row = list()
                scharry_row = list()
                for col_i, val in enumerate(row):
                    if col_i < self.pixels_frame[2] or col_i >= self.pixels_frame[3]:
                        continue

                    cell_f_val = self.add_cell_data(idx, row_i, col_i)
                    cur_pic_final_row.append(np.array(cell_f_val['cell_list']))
                    # build matrix of features to see their importance
                    if 'canny' in self.feature_method_list:
                        canny_row.append(cell_f_val['canny_val'])
                    if 'sobelx' in self.feature_method_list:
                        sobelx_row.append(cell_f_val['sobelx_val'])
                    if 'sobely' in self.feature_method_list:
                        sobely_row.append(cell_f_val['sobely_val'])
                    if 'laplacian' in self.feature_method_list:
                        laplacian_row.append(cell_f_val['laplacian_val'])
                    if 'scharrx' in self.feature_method_list:
                        scharrx_row.append(cell_f_val['scharrx_val'])
                    if 'scharry' in self.feature_method_list:
                        scharry_row.append(cell_f_val['scharry_val'])

                cur_pic_final_format.append(np.array(cur_pic_final_row))
                cur_pic_canny_final_format.append(np.array(canny_row))
                cur_pic_sobelx_final_format.append(np.array(sobelx_row))
                cur_pic_sobely_final_format.append(np.array(sobely_row))
                cur_pic_laplacian_final_format.append(np.array(laplacian_row))
                cur_pic_scharrx_final_format.append(np.array(scharrx_row))
                cur_pic_scharry_final_format.append(np.array(scharry_row))

                row_i += 1
            self.X.append(np.array(cur_pic_final_format))
            self.X_canny_final.append(np.array(cur_pic_canny_final_format))
            self.X_sobelx_final.append(np.array(cur_pic_sobelx_final_format))
            self.X_sobely_final.append(np.array(cur_pic_sobely_final_format))
            self.X_laplacian_final.append(np.array(cur_pic_laplacian_final_format))
            self.X_scharrx_final.append(np.array(cur_pic_scharrx_final_format))
            self.X_scharry_final.append(np.array(cur_pic_scharry_final_format))

        logging.info('finish merge features - X feature is ready')
        return

    def add_cell_data(self, idx, row_i, col_i):
        cell_list = list()
        canny_val = ''
        sobelx_val = ''
        sobely_val = ''
        laplacian_val = ''
        scharrx_val = ''
        scharry_val = ''

        if 'canny' in self.feature_method_list:
            canny_val = self.X_canny[idx][row_i][col_i]
            cell_list.append(canny_val)
        if 'sobelx' in self.feature_method_list:
            sobelx_val = self.X_sobelx[idx][row_i][col_i]
            cell_list.append(sobelx_val)
        if 'sobely' in self.feature_method_list:
            sobely_val = self.X_sobely[idx][row_i][col_i]
            cell_list.append(sobely_val)
        if 'laplacian' in self.feature_method_list:
            laplacian_val = self.X_laplacian[idx][row_i][col_i]
            cell_list.append(laplacian_val)
        if 'scharrx' in self.feature_method_list:
            scharrx_val = self.X_scharrx[idx][row_i][col_i]
            cell_list.append(scharrx_val)
        if 'scharry' in self.feature_method_list:
            scharry_val = self.X_scharry[idx][row_i][col_i]
            cell_list.append(scharry_val)

        # print(cell_list)
        return {
            'cell_list': cell_list,
            'canny_val': canny_val,
            'sobelx_val': sobelx_val,
            'sobely_val': sobely_val,
            'laplacian_val' :laplacian_val,
            'scharrx_val': scharrx_val,
            'scharry_val': scharry_val,
        }

    def add_sobelx_feature(self):
        import cv2
        from matplotlib import pyplot as plt
        for suffix_name in self.images_name_list:
            cur_path = os.path.join(self.images_dir, suffix_name) + '.jpg'

            # load as greyscale
            img = cv2.imread(cur_path, 0)
            # # load as RGB and convert to greyscale
            # img = self.images_pixel_dict[suffix_name]
            # img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
            edges = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
            if self.abs_flag:
                edges = np.absolute(edges) + self.kepsilon
            if self.log_flag:
                edges = np.log(edges)

            edges = cv2.bilateralFilter(edges, d=5, sigmaColor=50, sigmaSpace=50)
            # self.X.append(edges)

            # # transformation to [0, 255]
            # min = edges.min()
            # edges = edges - min
            # max = edges.max()
            # edges = edges/float(max) * 255
            # # heuristic transformation to binary
            # threshold1 = 130
            # edges[edges > threshold] = 255
            # edges[edges <= threshold] = 0

            # mean = np.mean(edges)
            # threshold2 = 1.5
            # edges[edges <= mean * threshold2] = 0
            # edges[edges > mean * threshold2] = 255

            # # logical splitting
            # threshold3a = 50
            # threshold3b = 204
            # edges[np.logical_or(edges <= threshold3a, edges >= threshold3b)] = 255
            # edges[np.logical_and(edges > threshold3a, edges < threshold3b)] = 0

            self.X_sobelx.append(edges)
            plot_edges = False
            if plot_edges:
                plt.subplot(121), plt.imshow(img, cmap='gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
        return

    def add_sobely_feature(self):
        import cv2
        from matplotlib import pyplot as plt
        for suffix_name in self.images_name_list:
            cur_path = os.path.join(self.images_dir, suffix_name) + '.jpg'
            img = cv2.imread(cur_path, 0)
            edges = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
            if self.abs_flag:
                edges = np.absolute(edges) + self.kepsilon
            if self.log_flag:
                edges = np.log(edges)

            edges = cv2.bilateralFilter(edges, d=5, sigmaColor=50, sigmaSpace=50)

            self.X_sobely.append(edges)
            plot_edges = False
            if plot_edges:
                plt.subplot(121), plt.imshow(img, cmap='gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
        return

    def add_laplacian_feature(self):
        import cv2
        from matplotlib import pyplot as plt
        for suffix_name in self.images_name_list:
            cur_path = os.path.join(self.images_dir, suffix_name) + '.jpg'
            img = cv2.imread(cur_path, 0)
            edges = cv2.Laplacian(img, cv2.CV_32F)
            if self.abs_flag:
                edges = np.absolute(edges) + self.kepsilon
            if self.log_flag:
                edges = np.log(edges)

            edges = cv2.bilateralFilter(edges, d=5, sigmaColor=50, sigmaSpace=50)

            self.X_laplacian.append(edges)
            plot_edges = False
            if plot_edges:
                plt.subplot(121), plt.imshow(img, cmap='gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
        return

    def add_canny_feature(self):
        import cv2
        from matplotlib import pyplot as plt

        def auto_canny(image, sigma=0.33):
            # compute the median of the single channel pixel intensities
            v = np.median(image)

            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(image, lower, upper)
            # return the edged image
            return edged

        if self.auto_canny:
            logging.info("using auto Canny with sigma {}".format(self.auto_canny_sigma))
        else:
            logging.info("using regular Canny")

        for suffix_name in self.images_name_list:
            cur_path = os.path.join(self.images_dir, suffix_name) + '.jpg'
            img = cv2.imread(cur_path, 0)
            if self.auto_canny:
                edges = auto_canny(img, self.auto_canny_sigma)      # TODO: handle cases when grey_picture=Ttue and auto_canny=True
            else:
                edges = cv2.Canny(img, 150, 250)

            # normalize to [0,1] prediction
            edges[edges == 255] = 1

            self.X_canny.append(edges)
            plot_edges = False
            if plot_edges:
                plt.subplot(121), plt.imshow(img, cmap='gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
        return

    def add_scharrx_feature(self):
        import cv2
        from matplotlib import pyplot as plt
        for suffix_name in self.images_name_list:
            cur_path = os.path.join(self.images_dir, suffix_name) + '.jpg'
            img = cv2.imread(cur_path, 0)
            edges = cv2.Scharr(img, cv2.CV_32F, 1, 0)
            if self.abs_flag:
                edges = np.absolute(edges) + self.kepsilon
            if self.log_flag:
                edges = np.log(edges)

            edges = cv2.bilateralFilter(edges, d=5, sigmaColor=50, sigmaSpace=50)

            self.X_scharrx.append(edges)
            plot_edges = False
            if plot_edges:
                plt.subplot(121), plt.imshow(img, cmap='gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
        return

    def add_scharry_feature(self):
        import cv2
        from matplotlib import pyplot as plt
        for suffix_name in self.images_name_list:
            cur_path = os.path.join(self.images_dir, suffix_name) + '.jpg'
            img = cv2.imread(cur_path, 0)
            edges = cv2.Scharr(img, cv2.CV_32F, 0, 1)
            if self.abs_flag:
                edges = np.absolute(edges) + self.kepsilon
            if self.log_flag:
                edges = np.log(edges)

            edges = cv2.bilateralFilter(edges, d=5, sigmaColor=50, sigmaSpace=50)

            self.X_scharry.append(edges)
            plot_edges = False
            if plot_edges:
                plt.subplot(121), plt.imshow(img, cmap='gray')
                plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                plt.subplot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
        return

    def label_percentage_valid_threshold(self, file_name):
        t = self.label_threshold
        import scipy.io
        mat_file_name = file_name + '.mat'
        pic_path = os.path.join(self.ground_truth_dir, mat_file_name)
        mat = scipy.io.loadmat(pic_path)
            # row_range = range(11, 17)
        row_range = range(self.pixels_frame[0], self.pixels_frame[1])
        column_range = range(self.pixels_frame[2], self.pixels_frame[3])
        all_picture = mat['groundTruth'][0][0]['Boundaries'][0][0]
        pic_after_cut = all_picture[row_range, :][:, column_range]

        # calculate percentage
        if np.average(pic_after_cut) > self.label_threshold:
            return True
        else:
            return False

def main():
    print('start create feature data set basic model')
    main_obj = FeatureDataSet()
    main_obj.extract_feature()
    print('finish LBP basic model')

if __name__ == "__main__":
    main()
