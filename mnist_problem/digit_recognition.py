from sklearn.datasets.mldata import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt


def display_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig("digit_image.png")


def main():
    mnist = fetch_mldata("MNIST original", data_home="/Users/anjaria.janit/scikit_learn_data/")
    X, y = mnist['data'], mnist['target']
    print 'Shape of X: ', X.shape
    print 'Shape of y: ', y.shape
    display_digit(X[36000])


main()
