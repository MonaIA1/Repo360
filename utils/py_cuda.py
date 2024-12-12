"""
first combile lib_mdbnet360.cu: 
nvcc -std=c++11 --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_mdbnet360.cu
"""

import ctypes
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

SENSOR_W = 640  # perspective image dim
SENSOR_H = 480  
VOXEL_SHAPE = (240,144,240)


def get_segmentation_class_map():
    return np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11,
                                   11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11], dtype=np.int32)
def get_class_names():
    return ["ceil.", "floor", "wall ", "wind.", "chair", "bed  ", "sofa ", "table", "tvs  ", "furn.", "objs."]

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src/lib_mdbnet360.so'))


_lib.setup.argtypes = (ctypes.c_int,
              ctypes.c_int,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_float,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_float, # cam hight param
              ctypes.c_float, # cam go back param
              )

def lib_mdbnet360_setup(device=0, num_threads=1024, v_unit=0.02, v_margin=0.24,
                         f=518.8, sensor_w=640, sensor_h=480,
                         vox_shape=None,
                         debug=0, cam_h=1.0, cam_b=1.0):

    global _lib, VOXEL_SHAPE

    if vox_shape is not None:
        VOXEL_SHAPE = vox_shape



    _lib.setup(ctypes.c_int(device),
                  ctypes.c_int(num_threads),
                  ctypes.c_float(v_unit),
                  ctypes.c_float(v_margin),
                  ctypes.c_float(f),
                  ctypes.c_float(sensor_w),
                  ctypes.c_float(sensor_h),
                  ctypes.c_int(VOXEL_SHAPE[0]),
                  ctypes.c_int(VOXEL_SHAPE[1]),
                  ctypes.c_int(VOXEL_SHAPE[2]),
                  ctypes.c_int(debug),
                  ctypes.c_float(cam_h),
                  ctypes.c_float(cam_b)
               )


_lib.get_point_cloud.argtypes = (ctypes.c_float,
                          ctypes.c_void_p,
                          ctypes.c_void_p,
                          ctypes.c_int,
                          ctypes.c_int,
                         )

def get_point_cloud(depth_file, baseline):
    global _lib

    depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    height, width = depth_image.shape
    
    num_pixels = height * width

    point_cloud = np.zeros((height, width, 6), dtype=np.float32)
    
    plt.imshow(depth_image)

    _lib.get_point_cloud(ctypes.c_float(baseline),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      point_cloud.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(width),
                      ctypes.c_int(height)
                      )
    return point_cloud, depth_image


_lib.get_voxels.argtypes = (ctypes.c_void_p, # for depth image
                            ctypes.c_void_p,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_void_p,
                            ctypes.c_int,
                            ctypes.c_void_p,
                            ctypes.c_void_p, # for depth_mapping_3d
                            
                            )


def get_voxels(depth_image,point_cloud, point_cloud_shape, min_x, max_x, min_y, max_y, min_z, max_z, vol_number=1):               
    global _lib, VOXEL_SHAPE

    boundaries = np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32)
    vox_grid = np.zeros(VOXEL_SHAPE,dtype=np.uint8)
    ##################
    # calculate the number of points in the point cloud
    num_points = point_cloud_shape[0] * point_cloud_shape[1] 
    num_voxels = VOXEL_SHAPE[0] * VOXEL_SHAPE[1] * VOXEL_SHAPE[2]
    depth_mapping = np.ones(num_voxels, dtype=np.float32) * (-1)
    
    ##################
    _lib.get_voxels(depth_image.ctypes.data_as(ctypes.c_void_p), # added depth image
                    point_cloud.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(point_cloud_shape[1]),
                    ctypes.c_int(point_cloud_shape[0]),
                    boundaries.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(vol_number),
                    vox_grid.ctypes.data_as(ctypes.c_void_p),

                    #################################################
                    depth_mapping.ctypes.data_as(ctypes.c_void_p),
                    
                    #################################################
    )
    depth_mapping = depth_mapping.astype(np.int64)
    return vox_grid, depth_mapping


_lib.downsample_grid.argtypes = (ctypes.c_void_p,
                            ctypes.c_void_p
                            )

def downsample_grid(in_vox_grid):
    global _lib, VOXEL_SHAPE

    out_vox_grid = np.zeros((VOXEL_SHAPE[0]//4,VOXEL_SHAPE[1]//4,VOXEL_SHAPE[2]//4 ),dtype=np.uint8)

    _lib.downsample_grid(in_vox_grid.ctypes.data_as(ctypes.c_void_p),
                        out_vox_grid.ctypes.data_as(ctypes.c_void_p)
                      )
    return out_vox_grid


_lib.downsample_limits.argtypes = (ctypes.c_void_p,
                            ctypes.c_void_p
                            )

def downsample_limits(in_vox_grid):
    global _lib, VOXEL_SHAPE

    out_vox_grid = np.zeros((VOXEL_SHAPE[0]//4,VOXEL_SHAPE[1]//4,VOXEL_SHAPE[2]//4 ),dtype=np.uint8)

    _lib.downsample_limits(in_vox_grid.ctypes.data_as(ctypes.c_void_p),
                        out_vox_grid.ctypes.data_as(ctypes.c_void_p)
                      )
    return out_vox_grid

_lib.FTSDFDepth.argtypes = (ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_void_p,
                           ctypes.c_float,
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_void_p,
                            ctypes.c_int
                         )

def get_ftsdf(depth_image, vox_grid, min_x, max_x, min_y, max_y, min_z, max_z,
               baseline, vol_number=1):
    global _lib, VOXEL_SHAPE

    boundaries = np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.float32)
    height, width = depth_image.shape
    vox_tsdf = np.zeros(VOXEL_SHAPE,dtype=np.float32)
    vox_limits = np.zeros(VOXEL_SHAPE,dtype=np.uint8)

    _lib.FTSDFDepth(depth_image.ctypes.data_as(ctypes.c_void_p),
                   vox_grid.ctypes.data_as(ctypes.c_void_p),
                   vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                   vox_limits.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_float(baseline),
                   ctypes.c_int(width),
                   ctypes.c_int(height),
                   boundaries.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(vol_number)
                  )
    return vox_tsdf, vox_limits



