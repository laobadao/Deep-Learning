import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab  
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    index = 25
        # 
    plt.imshow(train_set_x_orig[index])
    pylab.show()
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    # y = [1], it's a 'cat' picture.
    A = np.array(
        [[[[[2,3],[4,5]],[[5,6],[6,7]]],
        [[[2,3],[4,5]],[[5,6],[6,7]]],
        [[[2,3],[4,5]],[[5,6],[6,7]]]],
        [[[[2,3],[4,5]],[[5,6],[6,7]]],
        [[[2,3],[4,5]],[[5,6],[6,7]]],
        [[[2,3],[4,5]],[[5,6],[6,7]]]]])
    print(A.shape)

    
    # (2, 3, 2, 2, 2)