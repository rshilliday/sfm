import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """

    :param n_cameras Integer. Number of cameras/images currently resected
    :param n_points: number of distinct 3D points that have been triangulated
    :param camera_indices: List. Value at ith position is index of camera that sees ith 2D point
    :param point_indices: List. Value at ith position is index of 3D point that sees ith 2D point
    """
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A

def project(points, camera_params, K):
    """
    Projects 3D points onto camera coordinates

    :param points: N x 3 List of 3D point coordinates
    :param camera_params: N x 12 List of 12D camera parameters (r1, ... r9, t1, t2, t3)
    :param K: Intrinsics matrix
    """
    points_proj = []

    for idx in range(len(camera_params)): # idx applies to both points and cam_params, they are = length vectors
        R = camera_params[idx][:9].reshape(3,3)
        rvec, _ = cv2.Rodrigues(R)
        t = camera_params[idx][9:]
        pt = points[idx]
        pt = np.expand_dims(pt, axis=0)
        pt, _ = cv2.projectPoints(pt, rvec, t, K, distCoeffs=np.array([]))
        pt = np.squeeze(np.array(pt))
        points_proj.append(pt)

    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals (ie reprojection error) for Bundle Adjustment.

    :param params: List of all parameters. First n_cameras*12 parameters are (r1, ..., r9, t1, t2, t3)
    for each resected camera. Remaining n_points*3 paramaters are (x, y, z) coord of each triangulated point
    :param n_cameras: Integer. # of resected cameras
    :param n_points: Integer. # of triangulated points
    :param camera_indices: List of indices of cameras viewing each 2D observation
    :param point_indices: List of indices of 3D points corresponding to each 2D observation
    :points_2d: 2D pixel coordinates of each observation by a camera of a 3D point
    :param K: Intrinsics matrix
    """
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()

def do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol):
    """
    Perform Bundle Adjustment on all currently resected images and all triangulated 3D points. Return updated
    values for camera poses and 3D point coordinates.

    :param points3d_with_views: List of Point3D_with_views objects.
    :param R_mats: Dict mapping index of resected cameras to their Rotation matrix
    :param t_vecs: Dict mapping index of resected cameras to their translation vector
    :param resected_imgs: List of indices of resected images
    :param keypoints: List of lists of cv2.Keypoint objects. keypoints[i] is list for image i.
    :param ftol: Tolerance for change in total reprojection error. Used so scipy.optimize.least_squares knows
    when to stop adjusting
    """
    point_indices = []
    points_2d = []
    camera_indices = []
    points_3d = []
    camera_params = []
    BA_cam_idxs = {} # maps from true cam indices to 'normalized' (i.e 11, 23, 31 maps to -> 0, 1, 2)
    cam_count = 0

    for r in resected_imgs:
        BA_cam_idxs[r] = cam_count
        camera_params.append(np.hstack((R_mats[r].ravel(), t_vecs[r].ravel())))
        cam_count += 1

    for pt3d_idx in range(len(points3d_with_views)):
        points_3d.append(points3d_with_views[pt3d_idx].point3d)
        for cam_idx, kpt_idx in points3d_with_views[pt3d_idx].source_2dpt_idxs.items():
            if cam_idx not in resected_imgs: continue
            point_indices.append(pt3d_idx)
            camera_indices.append(BA_cam_idxs[cam_idx])#append normalized cam idx
            points_2d.append(keypoints[cam_idx][kpt_idx].pt)
    if len(points_3d[0]) == 3: points_3d = np.expand_dims(points_3d, axis=0)

    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    camera_indices = np.array(camera_indices)
    points_3d = np.squeeze(points_3d)
    camera_params = np.array(camera_params)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', loss='linear', ftol=ftol, xtol=1e-12, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))

    adjusted_camera_params = res.x[:n_cameras * 12].reshape(n_cameras, 12)
    adjusted_points_3d = res.x[n_cameras * 12:].reshape(n_points, 3)
    adjusted_R_mats = {}
    adjusted_t_vecs = {}
    for true_idx, norm_idx in BA_cam_idxs.items():
        adjusted_R_mats[true_idx] = adjusted_camera_params[norm_idx][:9].reshape(3,3)
        adjusted_t_vecs[true_idx] = adjusted_camera_params[norm_idx][9:].reshape(3,1)
    R_mats = adjusted_R_mats
    t_vecs = adjusted_t_vecs
    for pt3d_idx in range(len(points3d_with_views)):
        points3d_with_views[pt3d_idx].point3d = np.expand_dims(adjusted_points_3d[pt3d_idx], axis=0)

    return points3d_with_views, R_mats, t_vecs
