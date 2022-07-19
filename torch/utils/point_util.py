import numpy as np
import logging
from pycg.isometry import Isometry
from joblib import parallel_backend
import transforms3d
import math
from numpy.random import RandomState
import torch
import torch.nn.functional as F


class ICPConfig:
    TYPE_POINT2POINT = 0
    TYPE_POINT2POINT_XY = 2
    TYPE_POINT2POINT_XY_NO_ROT = 3
    TYPE_POINT2PLANE = 10

    def __init__(self):
        self.trans_init = Isometry()
        self.max_iteration = 30
        self.max_corr_dist = 0.15
        self.type = self.TYPE_POINT2POINT
        # Stopping criteria
        self.relative_ratio_change = 1.0e-6
        self.relative_err_change = 1.0e-6

    def compute_transformation(self, source_pc, target_pc, target_normal=None):
        if self.type == self.TYPE_POINT2POINT:
            source_center = np.mean(source_pc, axis=0, keepdims=True)
            target_center = np.mean(target_pc, axis=0, keepdims=True)
            source_pc_n = source_pc - source_center
            target_pc_n = target_pc - target_center
            H = source_pc_n.T @ target_pc_n
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            t = -R @ source_center.T + target_center.T
            return Isometry.from_matrix(R, t.ravel())
        elif self.type == self.TYPE_POINT2POINT_XY:
            source_center = np.mean(source_pc[:, :2], axis=0, keepdims=True)
            target_center = np.mean(target_pc[:, :2], axis=0, keepdims=True)
            source_pc_n = source_pc[:, :2] - source_center
            target_pc_n = target_pc[:, :2] - target_center
            H = source_pc_n.T @ target_pc_n
            U, S, Vt = np.linalg.svd(H)
            R = np.identity(3)
            t = np.zeros((3, 1))
            R[:2, :2] = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R[:2, :2] = Vt.T @ U.T
            t[:2, :] = -R[:2, :2] @ source_center.T + target_center.T
            return Isometry.from_matrix(R, t.ravel())
        elif self.type == self.TYPE_POINT2POINT_XY_NO_ROT:
            source_center = np.mean(source_pc[:, :2], axis=0, keepdims=True)
            target_center = np.mean(target_pc[:, :2], axis=0, keepdims=True)
            t = np.zeros((3, 1))
            t[:2, :] = target_center.T - source_center.T
            return Isometry.from_matrix(np.identity(3), t.ravel())
        elif self.type == self.TYPE_POINT2PLANE:
            assert target_normal is not None
            # Gauss-Newton r(xi) = (T(xi) source - target) @ normal_target
            #   cross is from skew matrix dTp/dxi
            r = np.sum((source_pc - target_pc) * target_normal, axis=1)
            J = np.concatenate([target_normal, np.cross(source_pc, target_normal)], axis=1)
            JTr = J.T @ r
            JTJ = J.T @ J
            xi = np.linalg.solve(JTJ, -JTr)
            return Isometry.from_twist(xi)
        else:
            raise NotImplementedError


def iterative_closest_point(source_pc: np.ndarray, target_pc: np.ndarray, target_normal: np.ndarray = None,
                            config: ICPConfig = None):
    """
    In order to pull together originally far-away clouds. Set initial transformation to translation.
    :param source_pc: (N, 3) numpy array
    :param target_pc:
    :param target_normal: (optional)
    :param config:
    :return:
    """
    if config is None:
        config = ICPConfig()

    if target_normal is not None:
        assert target_pc.shape[0] == target_normal.shape[0]

    from sklearn.neighbors import NearestNeighbors
    target_tree = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    target_tree.fit(target_pc)

    final_trans = config.trans_init
    source_pc = final_trans @ source_pc
    source_inds, target_inds = None, None

    def compute_correspondence():
        nonlocal source_inds, target_inds
        with parallel_backend('loky'):
            dist, target_inds = target_tree.kneighbors(source_pc)
        source_inds = np.where(dist < config.max_corr_dist)[0]
        target_inds = target_inds[source_inds, 0]
        if source_inds.shape[0] < 3:
            return None, None
        i_err = np.mean(dist[source_inds])
        i_ratio = source_inds.shape[0] / source_pc.shape[0]
        return i_err, i_ratio

    inlier_err, inlier_ratio = compute_correspondence()
    if inlier_err is None:
        return None

    # Iteratively, find correspondence and compute optimal transformation.
    for iter_i in range(config.max_iteration):
        logging.debug(f"ICP Iteration {iter_i}: Inlier = {inlier_ratio * 100:.3f}%, Err = {inlier_err:.4f}.")
        update = config.compute_transformation(source_pc[source_inds], target_pc[target_inds],
                                               target_normal[target_inds] if target_normal is not None else None)
        final_trans = update.dot(final_trans)
        source_pc = update @ source_pc

        backup_err, backup_ratio = compute_correspondence()
        if backup_err is None:
            return None

        if abs(backup_err - inlier_err) < config.relative_err_change and \
            abs(backup_ratio - inlier_ratio) < config.relative_ratio_change:
            break
        inlier_err, inlier_ratio = backup_err, backup_ratio

    return final_trans


def uniform_sampling_from_mesh(vert: np.ndarray, tris: np.ndarray, n_sample=1024, random_state=None,
                               return_uvs=False):
    """
    (This takes ~4x longer time than Open3D's C++ version, which shares the same algorithm though.)
    So as long as you don't need fine-grained control, do not use this.
    :return: sampled points
    """
    if random_state is None:
        random_state = RandomState()

    vert_tris = vert[tris]      # (F, 3, 3)
    v0, v1, v2 = vert_tris[:, 0], vert_tris[:, 1], vert_tris[:, 2]
    surface_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)      # (F,)
    surface_area_cdf = np.cumsum(surface_area)      # (F+1,)

    sample_t = random_state.uniform(0.0, surface_area_cdf[-1], size=n_sample)
    sample_tri_ids = np.searchsorted(surface_area_cdf, sample_t) - 1

    r1_sqrt = np.sqrt(random_state.uniform(0., 1., size=(n_sample, 1)))
    r2 = random_state.uniform(0., 1., size=(n_sample, 1))
    sa = 1 - r1_sqrt
    sb = r1_sqrt * (1 - r2)
    sc = r1_sqrt * r2

    final_pos = sa * v0[sample_tri_ids] + sb * v1[sample_tri_ids] + sc * v2[sample_tri_ids]

    if return_uvs:
        return final_pos, [sample_tri_ids, sa, sb, sc]

    return final_pos


def furthest_sampling_cloud(pts: np.ndarray, k: int, output_indices=False, random_state=None):
    """
    Sampling point cloud P using furthest sampling. This runs on CPU and can be slow.
    For GPU version, please use pointnet2_utils.furthest_point_sample which takes in pytorch tensor

    :param pts: N x C numpy array
    :param k: number to be sampled
    :param output_indices: Output indices of chosen points
    :return: farthest_pts: N x C, furthest_indices
    """
    # if pts.shape[0] == k:
    #     if not output_indices:
    #         return pts
    #     else:
    #         return np.arange(0, k)
    if random_state is None:
        random_state = RandomState()

    def _calc_distances(p0):
        return ((p0[0:3] - pts[:, 0:3]) ** 2).sum(axis=1)

    furthest_indices = [random_state.randint(len(pts))]
    distances = _calc_distances(pts[furthest_indices[-1]])
    for i in range(1, k):
        furthest_indices.append(np.argmax(distances))
        distances = np.minimum(
            distances, _calc_distances(pts[furthest_indices[-1]]))

    if not output_indices:
        return pts[furthest_indices, :]
    else:
        return np.asarray(furthest_indices)


def pad_cloud(P: np.ndarray, n_in: int, use_fps=False, return_inds=False, random_state=None):
    """
    Pad or subsample 3D Point cloud to n_in (fixed) number of points
    :param P: N x C numpy array
    :param n_in: number of points to truncate
    :param use_fps: Use furthest point sampling when doing sub-sampling, otherwise random choice is used.
    :return: n_in x C numpy array
    """
    if random_state is None:
        random_state = RandomState()

    N = P.shape[0]
    # https://github.com/charlesq34/pointnet/issues/41
    if N > n_in:  # need to subsample
        if use_fps:
            choice = furthest_sampling_cloud(P, n_in, True, random_state)
        else:
            choice = random_state.choice(N, n_in, replace=False)
    elif N < n_in:  # need to pad by duplication
        if use_fps:
            ii = furthest_sampling_cloud(P, n_in - N, True, random_state)
        else:
            ii = random_state.choice(N, n_in - N)
        choice = np.concatenate([range(N), ii])
    else:
        choice = np.arange(N)

    if return_inds:
        return choice
    else:
        return P[choice, :]


def normalize_cloud(P: np.ndarray):
    """
    Align the center of point cloud to (0,0,0).
    Rotation and Scale is not modified since they are the nature of data.
    :param P: N x 3 numpy array
    :return: N x 3 numpy array
    """
    max_pos = np.max(P, axis=0)
    min_pos = np.min(P, axis=0)
    mid_pos = (max_pos + min_pos) / 2
    return P - mid_pos


def augment_cloud(Ps, args):
    raise Exception("This function should not be used because it is not randomness safe.")
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1 / args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot:
        angle_x = random.uniform(0, 2 * math.pi)
        angle_y = random.uniform(0, 2 * math.pi)
        angle_z = random.uniform(0, 2 * math.pi)
        Rx = transforms3d.axangles.axangle2mat([1, 0, 0], angle_x)
        Ry = transforms3d.axangles.axangle2mat([0, 1, 0], angle_y)
        Rz = transforms3d.axangles.axangle2mat([0, 0, 1], angle_z)
        R = np.dot(Rx, np.dot(Ry, Rz))
        M = np.dot(R, M)
    if args.pc_augm_mirror_prob > 0:  # mirroring x&z, not y
        if random.random() < args.pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < args.pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
    result = []
    for P in Ps:
        P[:, :3] = np.dot(P[:, :3], M.T)
        if args.pc_augm_jitter:
            sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
        result.append(P)
    return result


def orient_cloud(Ps, orient_target):
    """" Orient mean normal of points to orient_target """
    for p in Ps:
        assert p.shape[1] == 6

    norm_mean = np.mean(Ps[0][:, 3:6], axis=0)
    norm_mean /= np.linalg.norm(norm_mean)

    orient_angle = math.acos(np.dot(norm_mean, orient_target))
    orient_axis = np.cross(norm_mean, orient_target)
    orient_axis /= np.linalg.norm(orient_axis)

    R = transforms3d.axangles.axangle2mat(orient_axis, orient_angle)

    result = []
    for P in Ps:
        P[:, :3] = np.dot(P[:, :3], R.T)
        P[:, 3:6] = np.dot(P[:, 3:6], R.T)
        result.append(P)
    return result


def batch_ops_download(*data):
    is_torch = isinstance(data[0], torch.Tensor)
    is_batch = (len(data[0].shape) != 2)
    dev = data[0].device if is_torch else None

    if is_torch:
        data = [t.cpu().detach().numpy() if t is not None else None for t in data]

    if not is_batch:
        data = [np.expand_dims(t, axis=0) if t is not None else None for t in data]

    return [is_torch, dev, is_batch], data


def batch_ops_upload(ctx, *data):
    is_torch, dev, is_batch = ctx

    if not is_batch:
        data = [t[0] for t in data]

    if is_torch:
        data = [torch.from_numpy(t).to(dev) for t in data]

    return data


def cloud_feature_fpfh(pc, normal=None, normal_est_radius: float = 0.05, fpfh_radius: float = 0.1):
    ctx, (pc, normal) = batch_ops_download(pc, normal)

    pc_features = []
    for bid, ind_pc in enumerate(pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ind_pc.astype(float))
        if normal is None:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_est_radius, max_nn=30))
        else:
            pcd.normals = o3d.utility.Vector3dVector(normal[bid].astype(float))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100))
        pcd_fpfh = np.asarray(pcd_fpfh.data)
        pc_features.append(pcd_fpfh.T)
    pc_features = np.stack(pc_features, axis=0)
    pc_features = batch_ops_upload(ctx, pc_features)[0]

    return pc_features


def estimate_cloud_normal(pc, nn_radius=.05, nn_max_nn=30, orient=None):
    ctx, (pc, ) = batch_ops_download(pc)

    pc_normals = []
    for ind_pc in pc:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ind_pc.astype(float))
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=nn_radius, max_nn=nn_max_nn))
        if orient is not None:
            pcd.orient_normals_towards_camera_location(orient)
        pc_normals.append(np.asarray(pcd.normals))
    pc_normals = np.stack(pc_normals, axis=0)
    pc_normals = batch_ops_upload(ctx, pc_normals)[0]

    return pc_normals


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def cloud_feature_ppf(radius: float, nsample: int, xyz: torch.Tensor, normals: torch.Tensor):
    """Sample and group for xyz, dxyz and ppf features

    Args:
        radius(int): Radius of cluster for computing local features
        nsample: Maximum number of points to consider per cluster
        xyz: XYZ coordinates of the points
        normals: Corresponding normals for the points (required for ppf computation)

    Returns:
        Dictionary containing the following fields ['dxyz', 'ppf'].
    """

    B, N, C = xyz.shape

    from utils.pointnet2_utils import ball_query
    xyz = xyz.contiguous()
    idx = ball_query(radius, nsample, xyz, xyz).long()          # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)                        # (B, npoint, nsample, C)
    d = grouped_xyz - xyz.view(B, N, 1, C)       # d = p_r - p_i  (B, npoint, nsample, 3)

    ni = index_points(normals, idx)
    nr = normals[:, :, None, :]

    nr_d = angle(nr, d)
    ni_d = angle(ni, d)
    nr_ni = angle(nr, ni)
    d_norm = torch.norm(d, dim=-1)

    xyz_feat = d                                                 # (B, npoint, n_sample, 3)
    ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, n_sample, 4)

    return xyz_feat, ppf_feat


def propagate_features(source_pc, target_pc, source_feat, nk=3, batched=True):
    from utils.pointconv_util import index_points_group

    if not batched:
        source_pc = source_pc.unsqueeze(0)
        target_pc = target_pc.unsqueeze(0)
        source_feat = source_feat.unsqueeze(0)

    dist = torch.cdist(target_pc, source_pc)  # (B, N, M)
    dist, group_idx = torch.topk(dist, nk, dim=-1, largest=False, sorted=False)     # (B, N, K)

    # Shifted reciprocal function.
    w_func = 1 / (dist + 1.0e-6)
    # w_func = torch.ones_like(dist)
    weight = (w_func / torch.sum(w_func, dim=-1, keepdim=True)).unsqueeze(-1)  # (B, N, k, 1)
    # print(weight)

    # weight = F.softmax(-dist, dim=-1).unsqueeze(-1)

    sparse_feature = index_points_group(source_feat, group_idx)
    full_flow = (sparse_feature * weight).sum(-2)  # (B, N, C)

    if not batched:
        full_flow = full_flow[0]

    return full_flow


if __name__ == '__main__':
    import open3d as o3d
    from pycg import vis

    pcd0 = o3d.io.read_point_cloud("/home/huangjh/Program/Open3D/examples/test_data/ICP/cloud_bin_0.pcd")
    pcd1 = o3d.io.read_point_cloud("/home/huangjh/Program/Open3D/examples/test_data/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4],
                             [0.0, 0.0, 0.0, 1.0]])
    trans_init = Isometry.from_matrix(trans_init, ortho=True)

    vis.show_3d([vis.pointcloud(pcd0, ucid=0).transform(trans_init.matrix), vis.pointcloud(pcd1, ucid=1)])

    pc0 = np.asarray(pcd0.points)
    pc1 = np.asarray(pcd1.points)
    normal1 = np.asarray(pcd1.normals)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd0, pcd1, 0.02, trans_init.matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    print("Start ICP...")
    logging.basicConfig(level=logging.DEBUG)
    icp_config = ICPConfig()
    icp_config.trans_init = trans_init
    icp_config.type = ICPConfig.TYPE_POINT2PLANE
    t01 = iterative_closest_point(pc0, pc1, normal1, config=icp_config)
    print(t01)

    vis.show_3d([vis.pointcloud(pcd0, ucid=0).transform(t01.matrix), vis.pointcloud(pcd1, ucid=1)])
