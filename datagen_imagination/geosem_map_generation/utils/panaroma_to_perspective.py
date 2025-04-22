"""
This file is originally from the Equirec2Perspec repository:
https://github.com/fuenwang/Equirec2Perspec

Original Author(s): Fu-En Wang
Licensed under the MIT License.

Modifications may have been made in this version.
"""

import cv2
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, mode='bilinear'):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #
    THETA = u_deg
    PHI = v_deg
    FOV = fov_deg
    (height, width) = out_hw

    if mode == 'bilinear':
        mode = cv2.INTER_LINEAR
    elif mode == 'nearest':
        mode = cv2.INTER_NEAREST
    elif mode == 'cubic':
        mode = cv2.INTER_CUBIC
    else:
        raise NotImplementedError('unknown mode')

    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
    K_inv = np.linalg.inv(K)
    
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz) 
    XY = lonlat2XY(lonlat, shape=e_img.shape).astype(np.float32)
    persp = cv2.remap(e_img, XY[..., 0], XY[..., 1], mode, borderMode=cv2.BORDER_WRAP)

    return persp
