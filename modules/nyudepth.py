# =========== #
#  Libraries  #
# =========== #
from glob import glob


# ======= #
#  Class  #
# ======= #
class NYUDepth:
    def __init__(self):
        self.depth_sparse_filenames = None
        self.depth_gt_filenames = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def read(self):
        """Get filenames for the input/output images"""

        self.depth_sparse_filenames = sorted(glob('nyu_depth/*/*_depth_sparse.png'))
        self.depth_gt_filenames = sorted(glob('nyu_depth/*/*_depth.png'))

    def train_test_split(self):
        # Inputs
        self.X_train = sorted(glob('nyu_depth/training/*_depth_sparse.png'))
        self.X_test = sorted(glob('nyu_depth/testing/*_depth_sparse.png'))

        # Outputs
        self.Y_train = sorted(glob('nyu_depth/training/*_depth.png'))
        self.Y_test = sorted(glob('nyu_depth/testing/*_depth.png'))

    def summary(self, showFilenames=False):
        if (self.depth_sparse_filenames is not None) and (self.depth_gt_filenames is not None):
            print(len(self.depth_sparse_filenames))
            print(len(self.depth_gt_filenames))
        else:
            # TODO: Otimizar
            if showFilenames:
                print(self.X_train)
                print(len(self.X_train))
                print(self.X_test)
                print(len(self.X_test))
                print(self.Y_train)
                print(len(self.Y_train))
                print(self.Y_test)
                print(len(self.Y_test))
            else:
                print(len(self.X_train))
                print(len(self.X_test))
                print(len(self.Y_train))
                print(len(self.Y_test))
