import numpy as np
from scipy import misc
from PIL import Image
from arguments import Arguments
import os, os.path
import logging

class VisualizationFeature:

    def __init__(self, X_canny, X_sobelx, X_sobely, X_laplacian, X_scharrx, X_scharry,
                 y, feature_method_list, dir_visualization_name, images_name_list, visualisation_flag):
        # saved_args = locals()
        # Arguments.logArguments('FeatureDataSet', saved_args)
        self.X_canny = X_canny
        self.X_sobelx = X_sobelx
        self.X_sobely = X_sobely
        self.X_laplacian = X_laplacian
        self.X_scharrx = X_scharrx
        self.X_scharry = X_scharry
        self.y = y
        self.feature_method_list = feature_method_list
        self.dir_visualization_name = dir_visualization_name        # save
        self.images_name_list = images_name_list
        self.visualisation_flag = visualisation_flag

    def run(self):
        self.create_dir()
        self.show_images()
        return

    def show_images(self):
        from scipy import misc
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np

        for image_idx, numpy_array in enumerate(self.y):
            fig = plt.figure()
            if 'canny' in self.feature_method_list:
                a = fig.add_subplot(2, 4, 1)
                a.set_title('Canny')
                plt.imshow(self.X_canny[image_idx])
            if 'sobelx' in self.feature_method_list:
                a = fig.add_subplot(2, 4, 2)
                a.set_title('sobelx')
                plt.imshow(self.X_sobelx[image_idx])
            if 'sobely' in self.feature_method_list:
                a = fig.add_subplot(2, 4, 3)
                a.set_title('sobely')
                plt.imshow(self.X_sobely[image_idx])
            if 'laplacian' in self.feature_method_list:
                a = fig.add_subplot(2, 4, 4)
                a.set_title('laplacian')
                plt.imshow(self.X_laplacian[image_idx])
            if 'sharrx' in self.feature_method_list:
                a = fig.add_subplot(2, 4, 5)
                a.set_title('Scharrx')
                plt.imshow(self.X_scharrx[image_idx])
            if 'scharry' in self.feature_method_list:
                a = fig.add_subplot(2, 4, 6)
                a.set_title('Scharry')
                plt.imshow(self.X_scharry[image_idx])

            # add ground truth always
            a = fig.add_subplot(2, 4, 7)
            a.set_title('Ground Truth')
            plt.imshow(self.y[image_idx])

            plt.savefig(self.vis_in_dir + str(self.images_name_list[image_idx]) + '.png', bbox_inches='tight')

            # only in our computers and flag is True show plots
            if os.name == 'nt' and self.visualisation_flag:
                plt.show()

            plt.close()
            # print(image_idx)
            # print(numpy_array)
        return

    def create_dir(self):
        if os.name == 'nt':
            vis_dir = '..\\visualisation_feature\\'
        elif os.name == 'posix':
            vis_dir = '../visualisation_feature/'

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if os.name == 'nt':
            vis_in_dir = '..\\visualisation_feature\\' + str(self.dir_visualization_name) + '\\'
        elif os.name == 'posix':
            vis_in_dir = '../visualisation_feature/' + str(self.dir_visualization_name) + '/'

        if not os.path.exists(vis_in_dir):
            os.makedirs(vis_in_dir)

        self.vis_in_dir = vis_in_dir
        return


def main():
    print('main is not supported')

if __name__ == "__main__":
    main()