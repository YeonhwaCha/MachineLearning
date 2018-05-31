import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_xy_histogram(H1, H2, B, min, max):
    plt.figure(figsize=(10, 5));
    opacity = 0.5
    [bincenters, binwidth] = np.linspace(min, max, num=B, retstep=True);
    rects1 = plt.bar(bincenters - (binwidth / 2), H1, binwidth,
                     alpha=opacity,
                     color='pink',
                     edgecolor='black',
                     label='Female')
    rects2 = plt.bar(bincenters + (binwidth / 2), H2, binwidth,
                     alpha=opacity,
                     color='b',
                     edgecolor='black',
                     label='Male')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.xticks(bincenters, bincenters.astype('int32'), fontsize=10)
    plt.legend()
    plt.show()

def plot_xyz_histogram(data1, data2):
    data_array_1 = np.array(data1)
    data_array_2 = np.array(data2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_data_1, y_data_1 = np.meshgrid(np.arange(data_array_1.shape[1]),
                                     np.arange(data_array_1.shape[0]))

    x_data_1 = x_data_1.flatten()
    y_data_1 = y_data_1.flatten()
    z_data_1 = data_array_1.flatten()

    x_data_2, y_data_2 = np.meshgrid(np.arange(data_array_2.shape[1]),
                                     np.arange(data_array_2.shape[0]))

    x_data_2 = x_data_2.flatten()
    y_data_2 = x_data_2.flatten()
    z_data_2 = data_array_2.flatten()

    ax.bar3d(x_data_1,
             y_data_1,
             np.zeros(len(z_data_1)),
             1, 1, z_data_1,
             color = 'pink',
             alpha = 0.3)
    ax.bar3d(x_data_2,
             y_data_2,
             np.zeros(len(z_data_2)),
             1, 1, z_data_2,
             color = 'blue',
             alpha = 0.3)

    plt.show()


def showMultipleTrainingSets(X, T):
    print('Checking multiple training vectors by plotting images.')
    plt.close('all')
    fig = plt.figure()
    nrows = 10
    ncols = 10
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows, ncols, row*ncols+col + 1)
            vectortoimg(X[np.random.randint(len(T))], show = False)
    plt.show()


def scatterplot(P, T, class1, class2):
    np.random.seed(0)
    randomorder = np.random.permutation(np.arange(len(T)))
    randomorder = np.arange(len(T))

    # Set colors
    cols = np.zeros((len(T), 4))  # Initialize matrix to hold colors
    cols[T == class1] = [1, 0, 0, 0.25]  # Negative points are red (with opacity 0.25)
    cols[T == class2] = [0, 1, 0, 0.25]  # Positive points are green (with opacity 0.25)

    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.scatter(P[randomorder, 1], P[randomorder, 0], s=5, linewidths=0, facecolors=cols[randomorder, :], marker="o");
    ax.set_aspect('equal')

    plt.gca().invert_yaxis()
    plt.show()


def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # the title of the figure
    # text( x(coordination), y(coordination), string(title), property )
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in xrange(len(images)):
        #add_subplot(row, columns, order_idx)
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        # if len(sptitles) == len(images):
        #     plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma', 10))
        # else:
        #     plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma', 10))

        plt.imshow(np.asarray(images[i]), cmap=colormap)

    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


def vectortoimg(v, show = True):
    plt.imshow(v.reshape(28, 28), interpolation = 'None', cmap = 'gray')
    plt.axis('off')
    if show:
        plt.show()