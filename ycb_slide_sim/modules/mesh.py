# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Mesh processing utilities 
"""

import itertools
import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial import KDTree
from modules.pose import pose_from_vertex_normal
from typing import Tuple
from scipy.spatial.transform import Rotation as R

# mesh utils
def sample_mesh(
    mesh: trimesh.base.Trimesh, num_samples: int, method: str = "even"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample mesh and return point/normals
    """
    sampled_points, faces = np.empty((0, 3)), np.array([], dtype=int)
    # https://github.com/mikedh/trimesh/issues/558 : trimesh.sample.sample_surface_even gives wrong number of samples
    while True:
        if method == "even":
            sP, f = trimesh.sample.sample_surface_even(mesh, count=num_samples)
        else:
            sP, f = trimesh.sample.sample_surface(mesh, count=num_samples)
        sampled_points = np.vstack([sampled_points, sP])
        faces = np.concatenate([faces, f])
        if len(sampled_points) <= num_samples:
            continue
        else:
            sampled_points, faces = sampled_points[:num_samples, :], faces[:num_samples]
            break

    sampled_normals = mesh.face_normals[faces, :]
    sampled_normals = sampled_normals / np.linalg.norm(sampled_normals, axis=1).reshape(
        -1, 1
    )
    return sampled_points, sampled_normals


def extract_edges(
    mesh: trimesh.base.Trimesh, num_samples: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract mesh edges via pyvista
    """
    mesh = pv.wrap(mesh)
    edges = mesh.extract_feature_edges(10)
    edges.compute_normals(inplace=True)  # this activates the normals as well

    tree = KDTree(mesh.points)
    _, ii = tree.query(edges.points, k=1)
    edgePoints, edgeNormals = edges.points, mesh.point_normals[ii, :]

    if edgePoints.shape[0] < num_samples:
        num_samples = edgePoints.shape[0]

    # https://stackoverflow.com/a/14262743/8314238
    indices = np.random.choice(edgePoints.shape[0], num_samples, replace=False)
    edgePoints = edgePoints[indices, :]
    edgeNormals = edgeNormals[indices, :] / np.linalg.norm(
        edgeNormals[indices, :], axis=1
    ).reshape(-1, 1)
    return edgePoints, edgeNormals, num_samples


def sample_mesh_edges(
    mesh: trimesh.base.Trimesh, num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample only mesh edges
    """
    sampled_edge_points, sampled_edge_normals, num_samples = extract_edges(
        mesh, num_samples
    )
    return sampled_edge_points, sampled_edge_normals


def sample_poses_on_mesh(
    mesh: trimesh.base.Trimesh,
    num_samples: int,
    edges: bool = True,
    constraint: np.ndarray = None,
    r: float = None,
    shear_mag: float = 5.0,
) -> np.ndarray:
    """
    Sample mesh and generates candidate sensor poses
    """
    if constraint is not None:
        constrainedSampledPoints, constrainedSampledNormals = np.empty(
            (0, 3)
        ), np.empty((0, 3))
        box = trimesh.creation.box(extents=[2 * r, 2 * r, 2 * r])
        box.apply_translation(constraint)
        constrainedMesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)
        while constrainedSampledPoints.shape[0] < num_samples:
            sP, sN = sample_mesh(constrainedMesh, num_samples * 100, method="even")
            dist = (np.linalg.norm(sP - constraint, axis=1)).squeeze()
            constrainedSampledPoints = np.append(
                constrainedSampledPoints, sP[np.less(dist, r), :], axis=0
            )
            constrainedSampledNormals = np.append(
                constrainedSampledNormals, sN[np.less(dist, r), :], axis=0
            )
        idxs = np.random.choice(constrainedSampledPoints.shape[0], num_samples)
        sampled_points, sampled_normals = (
            constrainedSampledPoints[idxs, :],
            constrainedSampledNormals[idxs, :],
        )
    elif edges:
        numSamplesEdges = int(0.3 * num_samples)
        sampled_edge_points, sampled_edge_normals, numSamplesEdges = extract_edges(
            mesh, numSamplesEdges
        )
        numSamplesEven = num_samples - numSamplesEdges
        sampledPointsEven, sampledNormalsEven = sample_mesh(mesh, numSamplesEven)
        sampled_points, sampled_normals = np.concatenate(
            (sampledPointsEven, sampled_edge_points), axis=0
        ), np.concatenate((sampledNormalsEven, sampled_edge_normals), axis=0)
    else:
        sampled_points, sampled_normals = sample_mesh(
            mesh, num_samples, method="normal"
        )

    # apply random pen into manifold
    shear_mag = np.radians(shear_mag)
    delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(num_samples,))
    T = pose_from_vertex_normal(sampled_points, sampled_normals, shear_mag, delta)
    return T


def sample_poses_on_mesh_minkloc(
    mesh: trimesh.base.Trimesh,
    num_samples: int,
    edges: bool = True,
    num_angles: int = 1,
    shear_mag: float = 5.0,
) -> np.ndarray:
    """
    Sample mesh and generates candidate sensor poses, custom for minkloc data
    """
    if edges:
        numSamplesEdges = int(0.3 * num_samples)
        sampled_edge_points, sampled_edge_normals, numSamplesEdges = extract_edges(
            mesh, numSamplesEdges
        )
        numSamplesEven = num_samples - numSamplesEdges
        sampledPointsEven, sampledNormalsEven = sample_mesh(mesh, numSamplesEven)
        sampled_points, sampled_normals = np.concatenate(
            (sampledPointsEven, sampled_edge_points), axis=0
        ), np.concatenate((sampledNormalsEven, sampled_edge_normals), axis=0)
    else:
        sampled_points, sampled_normals = sample_mesh(mesh, num_samples)

    sampled_points = np.repeat(sampled_points, num_angles, axis=0)
    sampled_normals = np.repeat(sampled_normals, num_angles, axis=0)

    # apply random pen into manifold
    delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(num_samples * num_angles,))
    T = pose_from_vertex_normal(sampled_points, sampled_normals, shear_mag, delta)
    return T

def sample_grid(mesh: trimesh.base.Trimesh, shear_mag: float = 5.0,)-> np.ndarray:
    z_max = mesh.vertices[:,2].max()
    org = np.array([0.0, 0.0, z_max])
    # xy_vals = np.array([-3,0,3])/1000.0
    xy_vals = [0.0]
    z_vals = -np.linspace(0,2,10) / 1000.0
    rot_vals = [-90., -45., 0., 45., 90.]
    tmp = [xy_vals, xy_vals, z_vals, rot_vals]
    offset_vals = list(itertools.product(*tmp))


    T = np.zeros((len(offset_vals),4,4))
    for i, offset in enumerate(offset_vals):
        T[i] = np.eye(4)
        r = R.from_euler('z', offset[-1], degrees=True)
        T[i][0:3,-1] = offset[0:3] + org
        T[i][0:3, 0:3] = r.as_matrix()
    return T, offset_vals

# def sample_closest_point(mesh: trimesh.base.Trimesh, p:np.ndarray):
#     new_p = np.zeros_like(p)
#     sP, f = trimesh.sample.sample_surface(mesh, count=10000)
#     for i, p0 in enumerate(p):
#         pose_t2 = p0[0:3,-1]
#         pose_t1= p[i-1][0:3,-1] if i>0 else pose_t2
#         pose_t0 = p[i-2][0:3,-1] if i>1 else pose_t2
#         a = np.linalg.norm(sP-pose_t2, axis=1)
#         b = np.linalg.norm(sP-pose_t1, axis=1)
#         c = np.linalg.norm(sP-pose_t0, axis=1)
#         idx = np.argmin(a+b+c)
#         new_p[i][0:3, 0:3] = p0[0:3, 0:3]
#         new_p[i][0:3,-1] = np.array(sP[idx])
#         new_p[i][-1,-1] = 1.0
#     return new_p

def sample_closest_point(mesh: trimesh.base.Trimesh, p:np.ndarray):
    points = p[:,0:3,-1]
    p_mod = p.copy()
    # prox = trimesh.proximity.ProximityQuery(mesh)
    # new_p, _, _ = prox.on_surface(p[:,0:3,-1])
    new_p, d, _  = trimesh.proximity.closest_point(mesh, points)
    p_mod[:,0:3,-1] = new_p

    # sP, f = trimesh.sample.sample_surface(mesh, count=2000)
    # for i, p0 in enumerate(p):
    #     if i==0:
    #         pose_t0 = p0[0:3,-1]
    #         idx = np.argmin(np.linalg.norm(sP-pose_t0, axis=1))
    #         new_p[i][0:3, 0:3] = p0[0:3, 0:3]
    #         new_p[i][0:3,-1] = np.array(sP[idx])
    #         new_p[i][-1,-1] = 1.0
    #     else:
    #         pose_t0 = p0[0:3,-1]
    #         new_pose = new_p[i-1][0:3,-1]
    #         a = np.linalg.norm(sP-pose_t0, axis=1)
    #         b = np.linalg.norm(sP-new_pose, axis=1)
    #         idx = np.argmin(a+b)
    #         new_p[i][0:3, 0:3] = p0[0:3, 0:3]
    #         new_p[i][0:3,-1] = np.array(sP[idx])
    #         new_p[i][-1,-1] = 1.0
    return p_mod