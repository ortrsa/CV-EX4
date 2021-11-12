import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    img_l = (img_l * 255.0).astype(numpy.uint8)
    img_r = (img_r * 255.0).astype(numpy.uint8)
    # L = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    # R = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    return disp(img_l, img_r, disp_range, k_size, SSD) / 255.0


def SSD(img_l, img_r, r, c, k_size, disp_range):
    """

    :param img_l:
    :param img_r:
    :param r:
    :param c:
    :param k_size:
    :param disp_range:
    :return:
    """

    ymin = 0
    _min = np.inf
    X = img_l[(r - k_size): (r + k_size), (c - k_size): (c + k_size)]

    for col in range(c - disp_range[1] // 2, c + disp_range[1] // 2):
        if np.abs(col - c) < disp_range[0] // 2:
            continue
        if col - k_size < 0:
            continue
        if col + k_size > img_r.shape[1]:
            break

        X_ = img_r[(r - k_size): (r + k_size), (col - k_size): col + k_size]

        _sum = np.sum((X - X_) ** 2)
        if _sum < _min:
            _min = _sum
            ymin = col

    return ymin


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    img_l = (img_l * 255.0).astype(numpy.uint8)
    img_r = (img_r * 255.0).astype(numpy.uint8)
    # L = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    # R = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    return disp(img_l, img_r, disp_range, k_size, NC)


def NC(img_l, img_r, r, c, k_size, disp_range):
    ymin = 0
    _max = 0
    X = img_l[(r - k_size): (r + k_size), (c - k_size): (c + k_size)]

    for col in range(c - disp_range[1] // 2, c + disp_range[1] // 2):
        if col - k_size < 0:
            continue
        if col + k_size > img_r.shape[1]:
            break
        X_ = img_r[(r - k_size): (r + k_size), (col - k_size): col + k_size]
        product = (X - np.mean(X)) * (X_ - np.mean(X_))
        stds = np.std(X) * np.std(X_)
        _sum = np.sum(product / stds)
        _sum /= X.size

        if _sum > _max:
            _max = _sum
            ymin = col

    return ymin


def disp(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int, Similarity_Measure):
    disp_map = np.zeros(img_l.shape)
    for r in range(k_size, img_l.shape[0] - k_size):
        for c in range(k_size, img_r.shape[1] - k_size):
            y = Similarity_Measure(img_l, img_r, r, c, k_size, disp_range)
            disp_map[r][c] = abs(y - c) / 255
    return disp_map


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """
    A = []
    for i in range(len(src_pnt)):
        x = src_pnt[i][0]
        y = src_pnt[i][1]
        x_ = dst_pnt[i][0]
        y_ = dst_pnt[i][1]

        A.append([x, y, 1, 0, 0, 0, -x_ * x, -x_ * y, -x_])
        A.append([0, 0, 0, x, y, 1, -y_ * x, -y_ * y, -y_])
    A = np.array(A)

    U, D, V_ = np.linalg.svd(A)
    V = V_[-1] / V_[-1, -1]
    M = V.reshape(3, 3)
    src_pnt = np.hstack((src_pnt, np.ones((src_pnt.shape[0], 1)))).T
    h_src = M.dot(src_pnt)
    h_src /= h_src[2, :]
    E = np.sqrt(np.sum(h_src[0:2, :] - dst_pnt.T) ** 2)

    return M, E


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and
       transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    src_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    def onclick_2(event):
        x_ = event.xdata
        y_ = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x_, y_))

        plt.plot(x_, y_, '*r')
        dst_p.append([x_, y_])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.gray()
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    fig2 = plt.figure()
    # display image 2
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.gray()
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    M, m = cv2.findHomography(src_p, dst_p)

    proj = np.zeros(dst_img.shape)
    mask = np.zeros(dst_img.shape)
    for y in range(src_img.shape[0]):
        for x in range(src_img.shape[1]):
            xy = np.array([x, y, 1])
            h_src = M.dot(xy)
            h_src /= h_src[2]
            x_ = int(h_src[0])
            y_ = int(h_src[1])

            # if the dst_img is out of bound continue
            try:
                proj[y_, x_] = src_img[y, x]
                mask[y_, x_] = 1
            except:
                continue

    n_blend, im_blend = pyrBlend(dst_img,proj, mask, 4)
    plt.gray()
    plt.imshow(im_blend)
    plt.show()


# ******************* copy from EX3 *********************
# *******************************************************


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    lap = []
    exp_list = []
    kernel = cv2.getGaussianKernel(5, -1)
    g_pyr = gaussianPyr(img, levels)
    for i in range(1, len(g_pyr)):
        exp_list.append(gaussExpand(g_pyr[i], kernel))
        # crop expanded image if needed
        if g_pyr[i - 1].shape != exp_list[i - 1].shape:
            exim = exp_list[i - 1]
            exim = exim[0:g_pyr[i - 1].shape[0], 0:g_pyr[i - 1].shape[1]]
            exp_list[i - 1] = exim
        lap.append(g_pyr[i - 1] - exp_list[i - 1])
    lap.append(g_pyr[-1])
    return lap


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel = cv2.getGaussianKernel(5, -1)

    lap_pyr.reverse()
    image = lap_pyr[0]
    for i in range(1, len(lap_pyr)):
        expandedImg = gaussExpand(image, kernel)
        if lap_pyr[i].shape != expandedImg.shape:
            expandedImg = expandedImg[0:lap_pyr[i].shape[0], 0:lap_pyr[i].shape[1]]
        image = expandedImg + lap_pyr[i]
    return image


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    pyr = [img]
    # ksize - kernel size, should be odd and positive (3,5,...)
    # sigma - Gaussian standard deviation.
    # If it is non-positive, it is computed from ksize as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    gk = cv2.getGaussianKernel(5, -1)
    kernel = np.dot(gk, gk.T)
    for i in range(levels):
        img = cv2.filter2D(pyr[i], -1, kernel, borderType=cv2.BORDER_REPLICATE)
        # take every second pixel and add the new image to list
        img = img[::2, ::2]
        pyr.append(img)
    return pyr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    # if kernel is 1D
    if gs_k.shape[1] == 1 and gs_k.shape[0] != 1:
        gs_k = np.dot(gs_k, gs_k.T)

    h_size = img.shape[0] * 2
    w_size = img.shape[1] * 2
    # make zeros matrix for gray or colored image
    if len(img.shape) > 2:
        new_im = np.zeros((h_size, w_size, 3))
    else:
        new_im = np.zeros((h_size, w_size))
    # equals every second pix to the small image and conv with gs_k
    new_im[::2, ::2] = img
    new_im = (cv2.filter2D(new_im, -1, gs_k * 4))
    return new_im


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    # make the image same shape
    sml_imPadded = imagePadding(img_1, img_2)
    mask_Padded = imagePadding(img_1, mask)

    # make gaussianPyr from mask and laplaceian from img 1 and 2.
    Gm = gaussianPyr(mask_Padded, levels)
    La = laplaceianReduce(img_1, levels)
    Lb = laplaceianReduce(sml_imPadded, levels)
    Lc = []
    for i in range(len(Gm)):
        Lc.append(Lb[i] * Gm[i] + (1 - Gm[i]) * La[i])
    # expand the image's and mask together.
    BlendedImage = laplaceianExpand(Lc)
    Naiveblend = sml_imPadded * mask_Padded + (1 - mask_Padded) * img_1

    return Naiveblend, BlendedImage


def imagePadding(image_1, image_2):
    """
    padding the small image with zero's to fit the big image shape.
    :param image_1:
    :param image_2:
    :return:
    """

    if image_1.shape >= image_2.shape:
        big_im = image_1
        sml_im = image_2
    else:
        big_im = image_2
        sml_im = image_1

    sml_imPadd = np.zeros(big_im.shape)
    sml_imPadd[(big_im.shape[0] - sml_im.shape[0]) // 2:(big_im.shape[0] - sml_im.shape[0]) // 2 + sml_im.shape[0],
    (big_im.shape[1] - sml_im.shape[1]) // 2:(big_im.shape[1] - sml_im.shape[1]) // 2 + sml_im.shape[1]] = sml_im
    return sml_imPadd
