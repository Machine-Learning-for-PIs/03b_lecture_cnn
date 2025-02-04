import numpy as np
from PIL import Image
from scipy.signal import correlate2d
import matplotlib.pyplot as plt


if __name__ == "__main__":
    problem_image = np.array(Image.open("./python/waldo_snow.jpg")) 
    waldo = np.array(Image.open("./python/waldo_small.jpg"))

    plt.imshow(waldo)
    plt.show()

    plt.imshow(problem_image)
    plt.show()



    problem_image = np.mean(problem_image, -1) # [1300:1500, 2500:2700]
    waldo = np.mean(waldo, -1)


    mean = np.mean(problem_image)
    std = np.std(problem_image)
    problem_image = (problem_image - mean)/std
    waldo = (waldo - mean)/std

    # problem_image += np.random.randn(*problem_image.shape)*0.1
    # waldo += np.random.randn(*waldo.shape)*0.1

    # plt.imshow(problem_image)
    # plt.imshow(waldo)
    # plt.show()
    # waldo_row, waldo_col = waldo.shape

    conv_res = correlate2d(problem_image, waldo, mode='same', boundary='wrap')

    max = np.argmax(conv_res)
    idx = np.unravel_index(max, conv_res.shape)
    print(idx)
    plt.imshow(np.log(np.abs(conv_res)))
    plt.plot(idx[1], idx[0], 'x')
    plt.colorbar()
    plt.show()

    plt.imshow(problem_image)
    plt.show()

    print('stop')
