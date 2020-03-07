import cv2
import numpy as np

def find_features(n_imgs, imgset):
    """
    Reads in n_imgs images from a directory and returns lists of keypoints and
    descriptors each of dimensions (n_imgs x n_kpts).

    :param n_imgs: Number of images to read in and process.
    :param imgset: String. Name of image set to read in and get matches/keypoints for
    """
    images = []
    keypoints = []
    descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(n_imgs):
        if imgset == 'templering':
            img = cv2.imread(f'./datasets/templeRing/{i:02d}.png', cv2.IMREAD_GRAYSCALE)
            K = np.matrix('1520.40 0.00 302.32; 0.00 1525.90 246.87; 0.00 0.00 1.00')
        elif imgset == 'Viking':
            img = cv2.imread(f'./datasets/Viking/{i:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            K = np.matrix('523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00')
        else: raise ValueError('Need to pass in valid imgset name!')
        images.append(img)
        kp, des = sift.detectAndCompute(images[-1], None)
        keypoints.append(kp)
        descriptors.append(des)
    return images, keypoints, descriptors, K

def find_matches(matcher, keypoints, descriptors, lowes_ratio=0.7):
    """
    Performs kNN matching with k=2 and Lowes' ratio test to return a list of dimensions
    n_imgs x n_imgs where matches[i][j] is the list of cv2.DMatch objects for images i and j

    :param matcher: Either a cv2.FlannBasedMatcher or cv2.BFMatcher
    :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    :param descriptors: List of lists of 128 dim lists (for SIFT). descriptors[i] is list for image i.
    """
    matches = []
    n_imgs = len(keypoints)
    for i in range(n_imgs):
        matches.append([])
        for j in range(n_imgs):
            if j <= i: matches[i].append(None)
            else:
                match = []
                m = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                for k in range(len(m)):
                    try:
                        if m[k][0].distance < lowes_ratio*m[k][1].distance:
                            match.append(m[k][0])
                    except:
                        continue
                matches[i].append(match)
    return matches


def remove_outliers(matches, keypoints):
    """
    Calculate fundamental matrix between 2 images to remove incorrect matches.
    Return matches with outlier removed. Rejects matches between images if there are < 20

    :param matches: List of lists of lists where matches[i][j][k] is the kth cv2.Dmatch object for images i and j
    :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    """

    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue
            kpts_i = []
            kpts_j = []
            for k in range(len(matches[i][j])):
                kpts_i.append(keypoints[i][matches[i][j][k].queryIdx].pt)
                kpts_j.append(keypoints[j][matches[i][j][k].trainIdx].pt)
            kpts_i = np.int32(kpts_i)
            kpts_j = np.int32(kpts_j)
            F, mask = cv2.findFundamentalMat(kpts_i, kpts_j, cv2.FM_RANSAC, ransacReprojThreshold=3)
            if np.linalg.det(F) > 1e-7: raise ValueError(f"Bad F_mat between images: {i}, {j}. Determinant: {np.linalg.det(F)}")
            matches[i][j] = np.array(matches[i][j])
            if mask is None:
                matches[i][j] = []
                continue
            matches[i][j] = matches[i][j][mask.ravel() == 1]
            matches[i][j] = list(matches[i][j])

            if len(matches[i][j]) < 20:
                matches[i][j] = []
                continue

    return matches

def num_matches(matches):
    """ Used for debugging. Count matches before/after outlier removal """
    n_matches = 0
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            n_matches += len(matches[i][j])

    return n_matches

def print_num_img_pairs(matches):
    num_img_pairs = 0
    num_pairs = 0
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            num_pairs += 1
            if len(matches[i][j]) > 0: num_img_pairs += 1

    print(f"Number of img pairs is {num_img_pairs} out of possible {num_pairs}")

def create_img_adjacency_matrix(n_imgs, matches):
    """
    Returns a n_imgs x n_imgs matrix where if img_adjacency[i][j] = 1, the images[i] and images[j]
    have a sufficient number of matches, and are regarded as viewing the same scene.

    :param n_imgs: Integer. Total number of images to be used in reconstruction
    :param matches: List of lists of lists where matches[i][j][k] is the kth cv2.Dmatch object for images i and j
    """
    num_img_pairs = 0
    num_pairs = 0
    pairs = []
    img_adjacency = np.zeros((n_imgs, n_imgs))
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if j <= i: continue
            num_pairs += 1
            if len(matches[i][j]) > 0:
                num_img_pairs += 1
                pairs.append((i,j))
                img_adjacency[i][j] = 1

    list_of_img_pairs = pairs
    return img_adjacency, list_of_img_pairs
