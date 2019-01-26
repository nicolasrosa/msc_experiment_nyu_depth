# examples/Python/Tutorial/Basic/rgbd_nyu.py

# =========== #
#  Libraries  #
# =========== #
from open3d import *
import numpy as np
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# =========== #
#  Functions  #
# =========== #
def read_nyu_pgm(filename, byteorder='>'):
    """ This is special function used for reading NYU pgm format as it is written in big endian byte order."""
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


# ====== #
#  Main  #
# ====== #
if __name__ == "__main__":
    # ----- Input ----- #
    print("Reading NYUDepth Raw images...")
    # Open3D does not support ppm/pgm file yet. Not using read_image here.
    # MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
    color_raw = mpimg.imread("nyu_depth_raw/r-1316653580.484909-1316500621.ppm")
    depth_raw = read_nyu_pgm("nyu_depth_raw/d-1316653580.471513-1316138413.pgm")

    color = Image(color_raw)
    depth = Image(depth_raw)

    rgbd_image = create_rgbd_image_from_nyu_format(color, depth)
    print(rgbd_image)

    # ----- SubPlot ----- #
    plt.subplot(1, 2, 1)
    plt.title('NYU grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('NYU depth image')
    plt.imshow(rgbd_image.depth)
    plt.draw()
    plt.pause(0.001)

    # ----- PointCloud ---- #
    pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    draw_geometries([pcd])
