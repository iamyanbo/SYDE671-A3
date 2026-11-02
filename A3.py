# Yanbo Cheng
# 21287485

from pathlib import Path
import time

from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import cv2
import imageio.v3 as iio
import numpy as np
from skimage.color import rgb2gray

from harris import get_harris_corners


def computeH(im1_pts, im2_pts, show_equations=False):
    A = []
    b = []

    # Build equations
    for (x, y), (u, v) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.extend([u, v])

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if show_equations:
        print("A =\n", A)
        print("b =\n", b)

    if np.linalg.matrix_rank(A) < 8:
        return None
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.append(h, 1).reshape(3, 3)
    return H


def find_bounds(im, H):
    corners = np.array([[0, 0], [im.shape[1] - 1, 0], [0, im.shape[0] - 1], [im.shape[1] - 1, im.shape[0] - 1]])
    new_corners = np.array([H @ np.append(corner, 1) for corner in corners])
    # Normalize
    new_corners = new_corners[:, :2] / new_corners[:, 2:]

    x_min, y_min = np.floor(new_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(new_corners.max(axis=0)).astype(int)
    return x_min, y_min, x_max, y_max

def warpImageNearestNeighbor(im, H):
    height, width = im.shape[:2]
    x_min, y_min, x_max, y_max = find_bounds(im, H)
    output_height = y_max - y_min + 1
    output_width = x_max - x_min + 1
    warped = np.zeros((output_height, output_width) + im.shape[2:])
    H_inv = np.linalg.inv(H)

    # Inverse warp
    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            source = H_inv @ np.array([x + x_min, y + y_min, 1])
            sx, sy = source[:2] / source[2]
            sx, sy = int(np.floor(sx + 0.5)), int(np.floor(sy + 0.5))
            if 0 <= sx < width and 0 <= sy < height:
                warped[y, x] = im[sy, sx]
    return warped


def warpImageBilinear(im, H):
    x_min, y_min, x_max, y_max = find_bounds(im, H)
    output_height = y_max - y_min + 1
    output_width = x_max - x_min + 1
    warped = np.zeros((output_height, output_width) + im.shape[2:])
    H_inv = np.linalg.inv(H)
    height, width = im.shape[:2]

    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            source = H_inv @ np.array([x + x_min, y + y_min, 1])
            sx, sy = source[:2] / source[2]
            if 0 <= sx <= width - 1 and 0 <= sy <= height - 1:
                # Four neighbors
                x0, y0 = int(np.floor(sx)), int(np.floor(sy))
                x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
                dx, dy = sx - x0, sy - y0
                # Weighted average
                top = (1 - dx) * im[y0, x0] + dx * im[y0, x1]
                bottom = (1 - dx) * im[y1, x0] + dx * im[y1, x1]
                warped[y, x] = (1 - dy) * top + dy * bottom
    return warped


def createMosaic(im1, im2, H):
    warped1 = warpImageBilinear(im1, H)
    # Mark pixels from image 1
    mask1 = warpImageBilinear(np.ones_like(im1), H)
    height1, width1 = warped1.shape[:2]
    height2, width2 = im2.shape[:2]

    # Shared canvas
    x_min1, y_min1, x_max1, y_max1 = find_bounds(im1, H)
    x_min, y_min = min(x_min1, 0), min(y_min1, 0)
    x_max, y_max = max(x_max1, width2 - 1), max(y_max1, height2 - 1)
    mosaic = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 3))
    count = np.zeros_like(mosaic)

    # Add both images to the canvas
    y1, x1 = y_min1 - y_min, x_min1 - x_min
    mosaic[y1:y1 + height1, x1:x1 + width1] += warped1
    count[y1:y1 + height1, x1:x1 + width1] += mask1

    y2, x2 = -y_min, -x_min
    mosaic[y2:y2 + height2, x2:x2 + width2] += im2
    count[y2:y2 + height2, x2:x2 + width2] += 1

    # Average where both images overlap
    return mosaic / np.maximum(count, 1)


def createPanorama(images, homographies):
    bounds = [find_bounds(im, H) for im, H in zip(images, homographies)]
    x_min = min(bound[0] for bound in bounds)
    y_min = min(bound[1] for bound in bounds)
    x_max = max(bound[2] for bound in bounds)
    y_max = max(bound[3] for bound in bounds)
    mosaic = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 3))
    count = np.zeros_like(mosaic)

    for im, H, bound in zip(images, homographies, bounds):
        warped = warpImageBilinear(im, H)
        mask = warpImageBilinear(np.ones_like(im), H)
        height, width = warped.shape[:2]
        x, y = bound[0] - x_min, bound[1] - y_min
        mosaic[y:y + height, x:x + width] += warped
        count[y:y + height, x:x + width] += mask

    return mosaic / np.maximum(count, 1)


def ANMS(h, coords, num_points=500):
    ys, xs = coords
    points = np.column_stack((xs, ys))
    scores = h[ys, xs]
    radii = np.full(len(points), np.inf)

    # Nearest stronger corner
    for i in range(len(points)):
        x, y = points[i]
        stronger = points[scores[i] < 0.9 * scores]
        if len(stronger) > 0:
            distances = np.sqrt((stronger[:, 0] - x)**2 + (stronger[:, 1] - y)**2)
            radii[i] = distances.min()

    order = np.argsort(radii)[::-1]
    return points[order[:num_points]]


def gaussian_filter(image, kernel_size=5, sigma=-1, boundary='fill'):
    g = cv2.getGaussianKernel(kernel_size, sigma)
    # Apply the Gaussian vertically, then horizontally
    if image.ndim == 2:
        blurred = convolve2d(image, g, mode='same', boundary=boundary)
        return convolve2d(blurred, g.T, mode='same', boundary=boundary)

    # Filter each RGB channel separately
    copy_image = np.empty(image.shape, dtype=np.float64)
    for c in range(image.shape[2]):
        blurred = convolve2d(image[:, :, c], g, mode='same', boundary=boundary)
        copy_image[:, :, c] = convolve2d(blurred, g.T, mode='same', boundary=boundary)

    return copy_image


def extractDescriptors(gray, points):
    descriptors = []
    for x, y in points:
        # Blur the 40 x 40 patch
        patch = gray[y - 20:y + 20, x - 20:x + 20]
        patch = gaussian_filter(patch, kernel_size=13, sigma=2, boundary='symm')

        # Sample every fifth pixel
        patch = patch[::5, ::5]

        # Normalize
        patch = patch - patch.mean()
        std = patch.std()
        if std > 0:
            patch = patch / std
        descriptors.append(patch)

    return np.array(descriptors)


def matchDescriptors(descriptors1, descriptors2, threshold=0.5):
    matches = []
    if len(descriptors2) < 2:
        return np.empty((0, 2), dtype=int)

    for i, descriptor1 in enumerate(descriptors1):
        errors = []
        for descriptor2 in descriptors2:
            error = np.sum((descriptor1 - descriptor2)**2)
            errors.append(error)

        order = np.argsort(errors)
        best_index = order[0]
        second_index = order[1]
        best_error = errors[best_index]
        second_error = errors[second_index]

        if second_error == 0:
            continue

        # Ratio test
        ratio = best_error / second_error
        if ratio < threshold:
            matches.append([i, best_index])

    return np.array(matches, dtype=int).reshape(-1, 2)


def ransacHomography(points1, points2, iterations=1000, threshold=3):
    best_inliers = []
    for _ in range(iterations):
        sample = np.random.choice(len(points1), 4, replace=False)
        H = computeH(points1[sample], points2[sample])
        if H is None:
            continue

        # Test all matches
        inliers = []
        for i, (point1, point2) in enumerate(zip(points1, points2)):
            prediction = H @ np.append(point1, 1)
            prediction = prediction[:2] / prediction[2]
            error = np.sqrt(np.sum((prediction - point2)**2))
            if error < threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Refit using all inliers
    H = computeH(points1[best_inliers], points2[best_inliers])
    return H, np.array(best_inliers)


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    out_dir = project_dir / "homography_output"
    out_dir.mkdir(exist_ok=True)

    # 1.1
    im1 = iio.imread(data_dir / "campus/left.png", mode="RGB").astype(float) / 255
    im2 = iio.imread(data_dir / "campus/right.png", mode="RGB").astype(float) / 255

    # 1.2
    im1_pts = np.array([
        [433.6, 374.4], [484.8, 369.4], [533.4, 376.6], [560.9, 518.2],
        [520.0, 531.1], [567.5, 502.2], [452.2, 448.6], [462.5, 577.2],
    ], dtype=float)
    im2_pts = np.array([
        [40.7, 296.7], [97.1, 302.2], [143.2, 317.0], [152.7, 454.4],
        [113.2, 467.2], [158.5, 440.7], [48.1, 377.6], [42.0, 510.2],
    ], dtype=float)

    H_manual = computeH(im1_pts, im2_pts, show_equations=True)
    print("H =\n", H_manual)

    # 1.3
    im = im1
    start = time.perf_counter()
    imwarped_nn = warpImageNearestNeighbor(im, H_manual)
    nearest_time = time.perf_counter() - start
    start = time.perf_counter()
    imwarped_bil = warpImageBilinear(im, H_manual)
    bilinear_time = time.perf_counter() - start
    print(f"Campus warp: nearest {nearest_time:.2f}s, bilinear {bilinear_time:.2f}s")

    plt.figure(figsize=(12, 5))
    for i, (image, title) in enumerate([(im, "Original"), (imwarped_nn, "Nearest neighbor"),
                                       (imwarped_bil, "Bilinear")]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.clip(image, 0, 1))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "campus_warp_comparison.png", dpi=150)
    plt.show()
    plt.close()

    # Corner order: top-left, top-right, bottom-left, bottom-right
    rectifications = [
        ("rectification/campus_right_window.png",
         [[16.7, 7.8], [74.5, 10.5], [14.3, 94.8], [72.4, 97.6]], 140, 210),
        ("rectification/campus_left_window.png",
         [[20.4, 22.6], [85.4, 18.6], [28.4, 185.4], [91.4, 180.9]], 150, 360),
    ]

    for filename, corners, width, height in rectifications:
        im = iio.imread(data_dir / filename, mode="RGB").astype(float) / 255
        rectangle = np.array([[0, 0], [width - 1, 0],
                              [0, height - 1], [width - 1, height - 1]])
        H_rect = computeH(corners, rectangle)

        start = time.perf_counter()
        nearest = warpImageNearestNeighbor(im, H_rect)
        nearest_time = time.perf_counter() - start
        start = time.perf_counter()
        bilinear = warpImageBilinear(im, H_rect)
        bilinear_time = time.perf_counter() - start
        # Crop to the rectangle
        x_min, y_min, _, _ = find_bounds(im, H_rect)
        nearest = nearest[-y_min:height - y_min, -x_min:width - x_min]
        bilinear = bilinear[-y_min:height - y_min, -x_min:width - x_min]
        print(f"{filename}: nearest {nearest_time:.2f}s, bilinear {bilinear_time:.2f}s")

        plt.figure(figsize=(12, 4))
        for i, (image, title) in enumerate([(im, "Original"), (nearest, "Nearest neighbor"),
                                            (bilinear, "Bilinear")]):
            plt.subplot(1, 3, i + 1)
            plt.imshow(np.clip(image, 0, 1))
            if title == "Original":
                points = np.array(corners)
                plt.scatter(points[:, 0], points[:, 1], c="red", s=12)
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{Path(filename).stem}_rectification.png", dpi=150)
        plt.show()
        plt.close()

    # 1.4
    manual_mosaic = createMosaic(im1, im2, H_manual)
    iio.imwrite(out_dir / "part_A_mosaic.png", (np.clip(manual_mosaic, 0, 1) * 255).round().astype(np.uint8))

    plt.figure(figsize=(10, 6))
    for j, (im, points) in enumerate([(im1, im1_pts), (im2, im2_pts)]):
        plt.subplot(1, 2, j + 1)
        plt.imshow(im)
        plt.scatter(points[:, 0], points[:, 1], c="red", s=12)
        for number, (x, y) in enumerate(points, start=1):
            plt.text(x + 3, y, str(number), color="yellow", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "part_A_manual_points.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(15, 5))
    for j, (image, title) in enumerate([(im1, "Image 1"), (im2, "Image 2"),
                                        (manual_mosaic, "Mosaic")]):
        plt.subplot(1, 3, j + 1)
        plt.imshow(np.clip(image, 0, 1))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "part_A_sources_and_mosaic.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # 2.1
    left = iio.imread(data_dir / "parking/left.png", mode="RGB").astype(float) / 255
    middle = iio.imread(data_dir / "parking/middle.png", mode="RGB").astype(float) / 255
    right = iio.imread(data_dir / "parking/right.png", mode="RGB").astype(float) / 255
    gray_left = rgb2gray(left)
    gray_middle = rgb2gray(middle)
    gray_right = rgb2gray(right)

    h_left, coords_left = get_harris_corners(gray_left)
    h_middle, coords_middle = get_harris_corners(gray_middle)
    h_right, coords_right = get_harris_corners(gray_right)
    points_left = ANMS(h_left, coords_left)
    points_middle = ANMS(h_middle, coords_middle)
    points_right = ANMS(h_right, coords_right)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.8))
    for ax, title in zip(axes, ["Original", "Harris corners", "After ANMS"]):
        ax.imshow(left)
        ax.set_title(title)
        ax.axis("off")
    axes[1].scatter(coords_left[1], coords_left[0], s=3, c="red")
    axes[2].scatter(points_left[:, 0], points_left[:, 1], s=3, c="red")
    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.savefig(out_dir / "corners_1.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)

    # 2.2
    descriptors_left = extractDescriptors(gray_left, points_left)
    descriptors_middle = extractDescriptors(gray_middle, points_middle)
    descriptors_right = extractDescriptors(gray_right, points_right)

    plt.figure(figsize=(12, 2))
    for j in range(6):
        plt.subplot(1, 6, j + 1)
        plt.imshow(descriptors_left[j], cmap="gray", interpolation="nearest")
        plt.title(f"Feature {j + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "descriptors_1.png", dpi=150)
    plt.show()
    plt.close()

    # 2.3
    matches_left = matchDescriptors(descriptors_left, descriptors_middle)
    matches_right = matchDescriptors(descriptors_right, descriptors_middle)
    matched_left = points_left[matches_left[:, 0]]
    middle_for_left = points_middle[matches_left[:, 1]]
    matched_right = points_right[matches_right[:, 0]]
    middle_for_right = points_middle[matches_right[:, 1]]

    height1, width1 = left.shape[:2]
    height2, width2 = middle.shape[:2]
    canvas = np.zeros((max(height1, height2), width1 + width2, 3))
    canvas[:height1, :width1] = left
    canvas[:height2, width1:] = middle
    plt.figure(figsize=(12, 5))
    plt.imshow(canvas)
    for (x1, y1), (x2, y2) in zip(matched_left, middle_for_left):
        plt.plot([x1, x2 + width1], [y1, y2], "-o", markersize=2, linewidth=0.5)
    plt.title(f"Feature matches: {len(matches_left)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "matches_1.png", dpi=150)
    plt.show()
    plt.close()

    height1, width1 = right.shape[:2]
    height2, width2 = middle.shape[:2]
    canvas = np.zeros((max(height1, height2), width1 + width2, 3))
    canvas[:height1, :width1] = right
    canvas[:height2, width1:] = middle
    plt.figure(figsize=(12, 5))
    plt.imshow(canvas)
    for (x1, y1), (x2, y2) in zip(matched_right, middle_for_right):
        plt.plot([x1, x2 + width1], [y1, y2], "-o", markersize=2, linewidth=0.5)
    plt.title(f"Feature matches: {len(matches_right)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "matches_2.png", dpi=150)
    plt.show()
    plt.close()

    # 2.4
    np.random.seed(671)
    H_left, inliers_left = ransacHomography(matched_left, middle_for_left)
    np.random.seed(671)
    H_right, inliers_right = ransacHomography(matched_right, middle_for_right)
    print(f"Left: {len(inliers_left)} / {len(matches_left)} RANSAC inliers")
    print(f"Right: {len(inliers_right)} / {len(matches_right)} RANSAC inliers")
    print("Left H =\n", H_left)
    print("Right H =\n", H_right)

    images = [left, middle, right]
    transforms = [H_left, np.eye(3), H_right]
    automatic_mosaic = createPanorama(images, transforms)
    iio.imwrite(out_dir / "part_B_mosaic.png",
                (np.clip(automatic_mosaic, 0, 1) * 255).round().astype(np.uint8))

    plt.figure(figsize=(15, 5))
    for i, (im, title) in enumerate(zip(images + [automatic_mosaic],
                                      ["Left", "Middle", "Right", "Part B: automatic"])):
        plt.subplot(1, 4, i + 1)
        plt.imshow(np.clip(im, 0, 1))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "part_B_sources_and_mosaic.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure(figsize=(14, 6))
    for i, (im, title) in enumerate([
        (manual_mosaic, "Part A: manual, 2 images"),
        (automatic_mosaic, "Part B: automatic, 3 images"),
    ]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(np.clip(im, 0, 1))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "part_A_and_B.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
