import cv2
import matplotlib.pyplot as plt
import numpy as np

import features
import processor
import structure
from sol1 import quantize


def ellipsoid_fit(x, y, z, mode=''):
    """ Fit an ellipsoid to a cloud of points using linear least squares
        Based on Yury Petrov MATLAB algorithm: "ellipsoid_fit.m"
    """

    # X = point_data[:, 0]
    # Y = point_data[:, 1]
    # Z = point_data[:, 2]

    X = x
    Y = y
    Z = z

    # AlGEBRAIC EQUATION FOR ELLIPSOID, from CARTESIAN DATA
    if mode == '':  # 9-DOF MODE
        D = np.array([X * X + Y * Y - 2 * Z * Z,
                      X * X + Z * Z - 2 * Y * Y,
                      2 * X * Y, 2 * X * Z, 2 * Y * Z,
                      2 * X, 2 * Y, 2 * Z,
                      1 + 0 * X]).T

    elif mode == 0:  # 6-DOF MODE (no rotation)
        D = np.array([X * X + Y * Y - 2 * Z * Z,
                      X * X + Z * Z - 2 * Y * Y,
                      2 * X, 2 * Y, 2 * Z,
                      1 + 0 * X]).T

    # THE RIGHT-HAND-SIDE OF THE LLSQ PROBLEM
    d2 = np.array([X * X + Y * Y + Z * Z]).T

    # SOLUTION TO NORMAL SYSTEM OF EQUATIONS
    u = np.linalg.solve(D.T.dot(D), D.T.dot(d2))
    # chi2 = (1 - (D.dot(u)) / d2) ^ 2

    # CONVERT BACK TO ALGEBRAIC FORM
    if mode == '':  # 9-DOF-MODE
        a = np.array([u[0] + 1 * u[1] - 1])
        b = np.array([u[0] - 2 * u[1] - 1])
        c = np.array([u[1] - 2 * u[0] - 1])
        v = np.concatenate([a, b, c, u[2:, :]], axis=0).flatten()

    elif mode == 0:  # 6-DOF-MODE
        a = u[0] + 1 * u[1] - 1
        b = u[0] - 2 * u[1] - 1
        c = u[1] - 2 * u[0] - 1
        zs = np.array([0, 0, 0])
        v = np.hstack((a, b, c, zs, u[2:, :].flatten()))

    else:
        pass

    # PUT IN ALGEBRAIC FORM FOR ELLIPSOID
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    # FIND CENTRE OF ELLIPSOID
    centre = np.linalg.solve(-A[0:3, 0:3], v[6:9])

    # FORM THE CORRESPONDING TRANSLATION MATRIX
    T = np.eye(4)
    T[3, 0:3] = centre

    # TRANSLATE TO THE CENTRE, ROTATE
    R = T.dot(A).dot(T.T)

    # SOLVE THE EIGENPROBLEM
    evals, evecs = np.linalg.eig(R[0:3, 0:3] / -R[3, 3])

    # SORT EIGENVECTORS
    # i = np.argsort(evals)
    # evals = evals[i]
    # evecs = evecs[:, i]
    # evals = evals[::-1]
    # evecs = evecs[::-1]

    # CALCULATE SCALE FACTORS AND SIGNS
    radii = np.sqrt(1 / abs(evals))
    sgns = np.sign(evals)
    radii *= sgns

    return (centre, evecs, radii)


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

    return tripoints3d


def display_image(title, img, active=True):
    if active:
        cv2.imshow(title, img)
        cv2.moveWindow(title, 150, 120)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_fruit_with_holes(img):
    # Quantize image.
    quantized_img, _ = quantize(img.astype(np.float64) / 255, 1, 50)
    quantized_img = (np.clip(quantized_img, 0, 1) * 255).astype(np.uint8)

    # Threshold image into binary mask
    gray_img = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)
    display_image('initial mask', binary_img)

    return binary_img


def find_fruit(img):
    img_copy = img.copy()

    # Fix small probability of failure.
    binary_img = None
    for _ in range(100):
        binary_img = find_fruit_with_holes(img_copy)
        if np.sum(binary_img) / 255 <= 0.7 * binary_img.size:
            break

    # Find contours and remove noise.
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [max(contours, key=cv2.contourArea)]

    # Fill in holes.
    result = np.zeros(img.shape)
    cv2.fillPoly(result, pts=contours, color=(255, 255, 255))
    display_image('filled shape', result)

    return result.astype(bool), contours


def apply_mask_to_fruit(img, mask):
    masked_img = img.copy()
    masked_img[~mask] = 0

    grid = (mask.copy().astype(np.float64) * 255).astype(np.uint8)
    grid[::16, ::16] = 128
    grid[~mask] = 0

    y_axis, x_axis, _ = np.nonzero(grid == 128)

    for point in zip(x_axis, y_axis):
        masked_img = cv2.circle(masked_img, point, radius=2, color=(0, 0, 0), thickness=-1)

    display_image('masked_img', masked_img)

    return masked_img


if __name__ == '__main__':
    # Normal
    img1 = cv2.imread('images/apple/1.png')
    img2 = cv2.imread('images/apple/2.png')

    mask1, _ = find_fruit(img1)
    mask2, _ = find_fruit(img2)

    masked_img1 = apply_mask_to_fruit(img1, mask1)
    masked_img2 = apply_mask_to_fruit(img2, mask2)

    tripoints3d = run_example(masked_img1, masked_img2)
    results = ellipsoid_fit(tripoints3d[0], tripoints3d[1], tripoints3d[2])

    print("(centre, evecs, radii)=", results)

