import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from goph547lab00.arrays import (
    square_ones,
)
def arrays():
    # test creating a square array of ones
    A_np = np.ones((3, 3))
    A = square_ones(3)
    print(f"A_np:\n{A_np}")
    print(f"A:\n{A}")
    print()

    # Question B1:
    C = np.ones((3, 5))
    print(f"Question B1: (3, 5) array of ones\n{C}")
    print()

    # Question B2:
    D = np.ones((6, 3)) * np.nan
    print(f"Question B2: (6, 3) array of nans\n{D}")
    print()

    #Question B3:
    odd = np.array([(2 * k + 1) for k in range(22, 38)])
    odd = np.reshape(odd, (len(odd), 1))
    print(f"Question B3: column vector of odd numbers from 44 to 75\n{odd}")
    print()

    # Question B4:
    odd_sum = np.sum(odd)
    print(f"Question B4: sum of odd numbers from B3\n{odd_sum}")
    print()

    # Question B5:
    A = np.array([[5, 7, 2], [1, -2, 3], [4, 4, 4]])
    print(f"Question B5: produce a specific (3, 3) array\n{A}")
    print()

    # Question B6:
    B = np.eye(3)
    print(f"Question B6: produce an identity matrix of rank 3\n{B}")
    print()

    # Question B7:
    AB = A * B
    print(f"Question B7: element wise multiplication A * B\n{AB}")
    print()

    # Question B8:
    AdotB = A @ B
    print(f"Question B8: inner product or dot product A @ B\n{AdotB}")
    print()

    # Question B9:
    AcrsB = np.cross(A, B)
    print(f"Question B9: cross product A x B, use numpy.cross()\n{AcrsB}")
    print()

def images():
    # Question B10-11
    # load and plot the image
    img = np.asarray(Image.open("rock_canyon.jpg"))
    img_shp = img.shape
    print(f"Question B11: shape of rock_canyon.jpg image\n{img_shp}")
    print(f"The image has {img_shp[0]} rows, {img_shp[1]} columns, and {img_shp[2]} channels (R,G,B)")
    print()
    plt.imshow(img)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Image rock_canyon.jpg")
    plt.show()

    # Question B12
    # reload the image in grayscale
    img_gray = np.asarray(Image.open("rock_canyon.jpg").convert("L"))
    img_gray_shp = img_gray.shape
    print(f"Question B12: shape of rock_canyon.jpg image converted to grayscale\n{img_gray_shp}")
    print(f"The image has {img_gray_shp[0]} rows and {img_gray_shp[1]} columns. No third component, since gray is one channel.")
    print()
    plt.imshow(img_gray, cmap="gray")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Grayscale version of rock_canyon.jpg")
    plt.show()

    # Question B13
    # extract the small pillar
    small_gray_image = img_gray[150:240, 110:150]
    plt.imshow(small_gray_image, cmap="gray")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Grayscale of small pillar")
    plt.show()

    # load image
    img = np.asarray(Image.open("rock_canyon.jpg"))
    img_shp = img.shape

    # create figure
    fig = plt.figure()
 
    # plot the original image
    plt.subplot(2, 2, 2)
    plt.imshow(img)
    plt.axis("off")

    # plot the RGB and mean channels along the x-coordinate
    plt.subplot(2, 2, 4)
    plt.plot(np.mean(img[:, :, 0], axis=0), "-r", linewidth=1.0, label="red")
    plt.plot(np.mean(img[:, :, 1], axis=0), "-g", linewidth=1.0, label="green")
    plt.plot(np.mean(img[:, :, 2], axis=0), "-b", linewidth=1.0, label="blue")
    plt.plot(np.mean(img.mean(axis=2), axis=0), "-k", linewidth=2.0, label="mean RGB")

    plt.xlim((0, img_shp[1]))
    plt.ylim((0, 256))
    plt.xlabel("x")
    plt.ylabel("color intensity")
    plt.legend()
    plt.show()

    # plot the RGB and mean channels along the x-coordinate
    plt.subplot(2, 2, 1)
    plt.plot(np.mean(img[:, :, 0], axis=1), range(0, img_shp[0]), "-r", linewidth=1.0)
    plt.plot(np.mean(img[:, :, 1], axis=1), range(0, img_shp[0]), "-g", linewidth=1.0)
    plt.plot(np.mean(img[:, :, 2], axis=1), range(0, img_shp[0]), "-b", linewidth=1.0)
    plt.plot(np.mean(np.mean(img, axis=2), axis=1), range(0, img_shp[0]), "-k", linewidth=2.0)
    plt.ylim((img_shp[0], 0))
    plt.xlim((0, 256))
    plt.ylabel("y")
    plt.xlabel("color intensity")

    fig.legend(loc="lower left")
    plt.savefig("rock_canyon_RGB_summary.png")
    plt.show()

    # (BONUS) Replotting the grayscale image from the mean of RGB channels
    img_gray_mean = np.mean(img, axis=2)
    plt.figure()
    plt.imshow(img_gray_mean, cmap="gray")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Grayscale version of rock_canyon.jpg from mean(RGB)")
    plt.show()

    plt.close()

