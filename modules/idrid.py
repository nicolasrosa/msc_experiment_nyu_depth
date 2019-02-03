# =========== #
#  Libraries  #
# =========== #
from glob import glob


# ======= #
#  Class  #
# ======= #
class IDRID:
    def __init__(self):
        self.image_filenames = None
        self.gt_filenames = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def train_test_split(self):
        # Inputs
        self.X_train = sorted(glob('datasets/idrid/training/image_*.jpg'))
        self.X_test = sorted(glob('datasets/idrid/testing/image_*.jpg'))

        # Outputs
        self.Y_train = sorted(glob('datasets/idrid/training/gt_*.png'))
        self.Y_test = sorted(glob('datasets/idrid/testing/gt_*.png'))

    def summary(self, showFilenames=False):
        if (self.image_filenames is not None) and (self.gt_filenames is not None):
            print(len(self.image_filenames))
            print(len(self.gt_filenames))
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
