import numpy as np

class SVM(object):

    def __init__(self):
        pass

    """
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
      with an appended bias dimensino in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 0 in CIFAR-10)
    - W is the weight matrix (e. g. 10 x 3073 in CIFAR-10)
    """
    def loss_i(self, x, y, W):
        delta = 1.0
        scores = W.dot(x)
        correct_class_score = scores[y]
        num_of_class = W.shape[0]
        loss_i = 0.0

        for j in xrange(num_of_class):
            if j == y:
                pass
            else:
                loss_i += max(0, scores[j] - correct_class_score + delta)

        return loss_i

    """
    A faster half-vectorized implementation. half-vectorized 
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this fuction)
    """
    def loss_i_vectorized(self, x, y, W):
        delta = 1.0
        margins = np.maximum(0, x - y + delta)
        margins[y] = 0
        loss_i = np.sum(margins)
        return loss_i

    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single exampel the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    def loss(self, x, y, W):
        delta = 1.0
        scores = W.dot(x)  # W*X
        scores_y = np.ones(scores.shape) * scores[y, np.arange(0, scores.shape[1])]
        margins = np.maximum(0, scores - scores_y + delta)
        margins[y, np.arange(0, scores.shape[1])] = 0
        data_loss = np.mean(np.sum(margins, axis=1))
        return data_loss

    ###################################################################
    # loss function's input parameter :
    # x : training image datas
    # y : training label datas
    # W : weight matrix
    #
    # D : dimension of one image data
    # N : the number of image data
    # C : the number of Class
    #
    # example
    # x : N(50000) * D(32*32*3+1=3073)
    # y : N(50000) * 1
    # W : D(3073) * C(10::CIFAR-10)


    """
    The gradient is analytically using Calculus, 
    which allows us to derive a direct formula for hte gradient 
    that is also very fast to cumpute.
    """

    def svm_loss_vectorized(self, x, y, W, delta, lamda):
        loss = 0.0
        dW = np.zeros(W.shape)

        ##############################################################
        # Data_Loss
        scores = x.dot(W) # scores = f(x_i, W) (50000*10) the number of class * the number of images
        scores_y = np.ones([scores.shape[1], scores.shape[0]]) * scores[np.arange(0, scores.shape[0]), y]
        scores_y = np.transpose(scores_y)
        data_losses = np.maximum(0, scores - scores_y + delta)
        data_losses[np.arange(0, scores.shape[0]), y] = 0 # Don't count y_i
        data_loss = np.mean(np.sum(data_losses, axis = 1))

        # Regularization_Loss
        regularization_loss = 0.5 * lamda * np.sum(W*W)

        # Loss = Data_Loss + Regularization_Loss
        loss = data_loss + regularization_loss
        ##############################################################

        grad = np.zeros(scores.shape)

        L = scores - scores_y + delta

        # indicator function that is one if the condition inside is true or zero otherwise.
        L[L < 0] = 0
        L[L > 0] = 1
        L[np.arange(0, scores.shape[0]), y] = 0 # Don't count y_i
        # taking the gradient with respect to w_y_i :: sum of the losses of class in each image.
        #
        L[np.arange(0, scores.shape[0]), y] = -1 * np.sum(L, axis=1)
        dW = np.transpose(L).dot(x)

        # Average over number of training examples.
        the_num_of_train = x.shape[0]
        dW /= the_num_of_train

        return loss, dW