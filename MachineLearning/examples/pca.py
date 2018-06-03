from pre_processing.PCA import pca, normalize, project, reconstruct
from models.model_pca import EigenVectorModel
from utils.load_dataset import load_att
from utils.visualize import subplot

import matplotlib.cm as cm

if __name__  == "__main__":
    ######################################################################
    # Load DataSet
    # DataSet CopyRight : AT&T Laboratories Cambridge
    # http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    ######################################################################
    X, T, shape = load_att("../../DataSet/PCA/orl_faces")

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