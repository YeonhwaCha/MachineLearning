from pre_processing.PCA import pca, normalize, project, reconstruct
from models.model_pca import EigenVectorModel
from utils.load_dataset import load_dataset
from utils.visualize import subplot

import matplotlib.cm as cm

# TEST_SET = "att"
TEST_SET = "mnist"

if __name__  == "__main__":
    ######################################################################
    # Load DataSet
    # DataSet CopyRight : AT&T Laboratories Cambridge
    # http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    ######################################################################
    if TEST_SET == "att":
        file_path = "../../DataSet/PCA/orl_faces"
        X, T, shape = load_dataset(TEST_SET, file_path)
    ######################################################################
    # Load DataSet
    # DataSet CopyRight : New York Univ. Google Labs.
    # http://yann.lecun.com/exdb/mnist/
    ######################################################################
    else:
        file_path = "../../DataSet/PCA/mnist/"
        X, T, shape = load_dataset(TEST_SET, file_path, set = "training", selecteddigits = [2, 8])

    [eigen_value, eigen_vector, mean] = pca(X, T)

    ######################################################################
    # Reconstruction
    ######################################################################
    steps = [i for i in xrange(10, min(len(X), 320), 20)]

    E = []
    data_idx = 100
    for i in xrange(min(len(steps), 16)):
        numEvs = steps[i]
        P = project(eigen_vector[:, 0:numEvs], X[data_idx], mean)
        R = reconstruct(eigen_vector[:, 0:numEvs], P, mean)
        R = R.reshape(shape)
        E.append(normalize(R, 0, 255))

    subplot(title="Reconstruction", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename=None)

    ######################################################################
    # modeling
    ######################################################################
    # choose 210 as a principal componant.
    principal_componants = 210
    model = EigenVectorModel(X, T, principal_componants)

    ######################################################################
    # classification
    ######################################################################
    classifier = model.predict(X[data_idx])
    print "classification : {0}".format(classifier)