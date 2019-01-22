# =========== #
#  Libraries  #
# =========== #
import glob
from random import sample

import cv2
import imageio
import numpy as np
from scipy import misc

# showImages = True
showImages = False


# =========== #
#  Functions  #
# =========== #
def remove_pixels(remove_percentage, px_idx, src):
    n_remove_px = round(remove_percentage * src.shape[0] * src.shape[1])

    res = src.copy()
    px_list = sample(range(len(px_idx)), n_remove_px)

    # print(px_list)

    for px in np.asarray(px_idx)[px_list]:
        # print(px)
        # print(px[0], px[1])
        res[px[0], px[1]] = 0

    # input("aki")

    return res


# ====== #
#  Main  #
# ====== #
def main():
    images = sorted(glob.glob('nyu_depth/*/*_colors.png'))
    depths = sorted(glob.glob('nyu_depth/*/*_depth.png'))

    print(images)
    print(len(images))
    print(depths)
    print(len(depths))

    ex_image = misc.imread(images[0])
    ex_depth = misc.imread(depths[0]).astype(np.uint16)
    image_shape = ex_image.shape
    depth_shape = ex_depth.shape

    print("images[0].shape: {}, dtype: {}".format(image_shape, ex_image.dtype))
    print("depths[0].shape: {}, dtype: {}".format(depth_shape, ex_depth.dtype))
    print(np.min(ex_depth), np.max(ex_depth))
    print()

    px_idx = [[i, j] for i in range(image_shape[0]) for j in range(image_shape[1])]

    i = 1
    for image_filename, depth_filename in zip(images, depths):
        print("{}/{}".format(i, len(images)))
        image = misc.imread(image_filename)

        # depth = misc.imread(depth_filename).astype(np.uint16)/1000.0 # It's not necessary to doing this in this code!
        depth = misc.imread(depth_filename).astype(np.uint16)

        scaled_depth = cv2.convertScaleAbs(depth * (255 / np.max(depth)))  # Only for Visualization Purpose

        # Keeps only 10% of the Pixels
        # new_image = removePixels(0.3, px_idx, image)
        # The removed pixels for the saved depth and the displayed images are different!!!
        new_depth = remove_pixels(0.9, px_idx, depth)  #
        new_scaled_depth = remove_pixels(0.9, px_idx, scaled_depth)  #

        # Display Results
        if showImages:
            # print(image)
            # print(depth)
            cv2.imshow('image', image)
            cv2.imshow('scaled_depth', scaled_depth)
            cv2.imshow('new_scaled_depth', new_scaled_depth)

        # Saving
        imageio.imsave(depth_filename.replace('_depth.png', '_depth_sparse.png'), new_depth)

        i += 1

        k = cv2.waitKey(5000)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break

    print("Done")


if __name__ == '__main__':
    main()
