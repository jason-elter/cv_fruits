import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import features
import processor
import structure
from sol1 import quantize
import sol4


def generate_points(img1, img2, mask1=None, mask2=None):
    pts1, pts2 = features.find_correspondence_points(img1, img2, mask1, mask2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)

    fig, ax = plt.subplots(1, 2)
    ax[0].autoscale_view('tight')
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].plot(points1[0], points1[1], 'r.')
    ax[1].autoscale_view('tight')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].plot(points2[0], points2[1], 'r.')
    fig.show()

    height, width, ch = img1.shape
    intrinsic = np.array([
        [1180.73916, 10.1288603, 870.449618],
        [0, 1094.15109, 435.217448],
        [0, 0, 1]])

    return points1, points2, intrinsic


def run_example(img1, img2, mask1=None, mask2=None):
    points1, points2, intrinsic = generate_points(img1, img2, mask1, mask2)

    # Calculate essential matrix with 2d points.
    # Result will be up to a scale
    # First, normalize points
    points1n = np.dot(np.linalg.inv(intrinsic), points1)
    points2n = np.dot(np.linalg.inv(intrinsic), points2)

    E = structure.compute_essential_normalized(points1n, points2n)
    print('Computed essential matrix:', (-E / E[0][1]))

    # Given we are at camera 1, calculate the parameters for camera 2
    # Using the essential matrix returns 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = structure.reconstruct_one_point(
            points1n[:, 0], points2n[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    # tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()

    return P1, P2, (-E / E[0][1])


def display_image(title, img, active=True):
    if active:
        cv2.imshow(title, img)
        cv2.moveWindow(title, 150, 120)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_fruit(img):
    img_copy = img.copy()

    # Quantize image.
    quantized_img, _ = quantize(img_copy.astype(np.float64) / 255, 1, 50)
    quantized_img = (np.clip(quantized_img, 0, 1) * 255).astype(np.uint8)

    # Threshold image into binary mask
    gray_img = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
    display_image('initial mask', binary_img)

    # Find contours.
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill in holes.
    result = np.zeros(img.shape)
    cv2.fillPoly(result, pts=contours, color=(255, 255, 255))
    display_image('filled shape', result)

    return result, contours


if __name__ == '__main__':
    # Normal
    img1 = cv2.imread('images/lemon/lemon001.jpeg')
    img2 = cv2.imread('images/lemon/lemon002.jpeg')

    mask1, contours1 = find_fruit(img1)
    mask2, contours2 = find_fruit(img2)

    P1, P2, E = run_example(img1, img2)

    # img1_copy = img1.copy()
    # img2_copy = img2.copy()
    # contoured_img1 = cv2.drawContours(img1_copy, contours1, -1, (0, 0, 255), 3)
    # contoured_img2 = cv2.drawContours(img2_copy, contours2, -1, (0, 0, 255), 3)

    # name = "lemon"
    # ext = "jpeg"
    # panorama_gen = sol4.PanoramicVideoGenerator(os.path.join('images', '%s') % name, name, 2, ext)
    # panorama_gen.align_images(translation_only=False)
    #
    # warped_img1 = sol4.warp_image(img1, panorama_gen.homographies[0])
    # warped_img2 = sol4.warp_image(img2, panorama_gen.homographies[1])
    # display_image("warped1", warped_img1)
    # display_image("warped1", warped_img2)
    #
    # # tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)
    #
    # fig = plt.figure()
    # fig.suptitle('3D reconstructed', fontsize=16)
    # ax = fig.gca(projection='3d')
    # # ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    # ax.set_xlabel('x axis')
    # ax.set_ylabel('y axis')
    # ax.set_zlabel('z axis')
    # ax.view_init(elev=135, azim=90)
    # plt.show()

