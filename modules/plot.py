# ===========
#  Libraries
# ===========
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========
def updateColorBar(cbar, img):
    vmin, vmax = np.min(img), np.max(img)
    cbar.set_clim(vmin, vmax)

    cbar_ticks = np.linspace(vmin, vmax, num=7, endpoint=True)
    cbar.set_ticks(cbar_ticks)

    cbar.draw_all()

    # Debug
    # print("vmin:", vmin, "\tvmax:", vmax)


# ===================
#  Class Declaration
# ===================
class Plot(object):
    def __init__(self, mode, title):
        self.fig, self.axes = None, None

        if mode == 'train':  # and Validation
            self.fig, self.axes = plt.subplots(7, 1, figsize=(14, 3))
            self.axes[0] = plt.subplot(131)
            self.axes[1] = plt.subplot(132)
            self.axes[2] = plt.subplot(133)
            # self.axes[3] = plt.subplot(134)

            # Sets Titles
            self.axes[0].set_title("Eye")
            self.axes[1].set_title("Disc (GT)")
            self.axes[2].set_title("Disc (Pred)")

        elif mode == 'test':
            self.fig = plt.figure()
            # self.fig.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9)
            X = [(3, 5, (1, 2)),
                 (3, 5, (6, 7)),
                 (3, 5, 3),
                 (3, 5, 8),
                 (3, 5, 13),
                 (3, 5, (11, 12)),
                 (3, 5, 14),
                 (3, 5, 15)]
            self.axes = []
            for nrows, ncols, plot_number in X:
                self.axes.append(self.fig.add_subplot(nrows, ncols, plot_number))

            # Sets Titles
            self.axes[0].set_title("Image")
            self.axes[1].set_title("Depth")
            self.axes[2].set_title("Image Resized")
            self.axes[3].set_title("Depth Resized")
            self.axes[4].set_title("Pred")
            self.axes[5].set_title("up(Pred)")
            self.axes[6].set_title("Pred_50")
            self.axes[7].set_title("Pred_80")

        self.fig.canvas.set_window_title(title)
        self.fig.tight_layout(pad=0.1, w_pad=None, h_pad=None)  # Fix Subplots Spacing

        self.isInitialized = False

    def showTrainResults(self, raw, label, pred):

        if not self.isInitialized:
            self.cax0 = self.axes[0].imshow(raw)
            self.cax1 = self.axes[1].imshow(label)
            self.cax2 = self.axes[2].imshow(pred)

            # Creates ColorBars
            self.cbar0 = self.fig.colorbar(self.cax0, ax=self.axes[0])
            self.cbar1 = self.fig.colorbar(self.cax1, ax=self.axes[1])
            self.cbar2 = self.fig.colorbar(self.cax2, ax=self.axes[2])

            self.isInitialized = True
        else:
            # Updates Colorbars
            updateColorBar(self.cbar0, raw)
            updateColorBar(self.cbar1, label)
            updateColorBar(self.cbar2, pred)

            # Updates Images
            self.cax0.set_data(raw)
            self.cax1.set_data(label)
            self.cax2.set_data(pred)
            plt.draw()

        plt.pause(0.001)
