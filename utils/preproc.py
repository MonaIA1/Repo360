# By: https://gitlab.com/UnBVision/edgenet360/-/blob/master/lib_edgenet360/preproc.py?ref_type=heads

import numpy as np
from sklearn import linear_model
import cv2
import pandas as pd
import math

PI = 3.14159265
DISP_SCALE = 2.0
DISP_OFFSET = -120.0


def plane_estimate(fpc, a, b, c):

    X = np.ones((fpc.shape[0], 2))
    X[:,0] = fpc[:,a]
    X[:,1] = fpc[:,b]

    y = fpc[:,c]

    ransac = linear_model.RANSACRegressor()
    try:
        ransac.fit(X, y)
        return [ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.intercept_]
    except:
        return [1,1,1]



def adjust_ceil(point_cloud, candidate, margin):
    wx, wy, wz, lat, long, rd = tuple(range(6))

    points = ((point_cloud[:, :, wy] < (candidate + margin)) & (point_cloud[:, :, wy] > (candidate - margin)))

    fpc = point_cloud[points]

    fit = plane_estimate(fpc, wx, wz, wy)

    if (abs(fit[0])<0.05) and (abs(fit[1])<0.05):
        return fit[2]
    else:
        return candidate



def get_limits(values, perc=.9, lperc=.9999):
    min = np.percentile(values, lperc)
    max = -np.percentile(-values, lperc)
    #print(min,max)
    values = values[(values >= min) & (values <= max)]
    return np.percentile(values, perc), -np.percentile(-values, perc)


def find_limits_v2(point_cloud, perc=.9, lperc=.9999):
    wx, wy, wz = tuple(range(3))
    print("l-r")
    left, right = get_limits(point_cloud[:, :, wx].flatten(), perc=perc, lperc=lperc)
    print(left,right)
    print("b-f")
    back, front = get_limits(point_cloud[:, :, wz].flatten(), perc=perc, lperc=lperc)
    print(back,front)

    print("f-c")
    floor, ceil = get_limits(point_cloud[:, :, wy].flatten(), perc=perc, lperc=lperc)
    print(floor,ceil)
    return ceil, floor, front, back, right, left



def find_region(startx, starty, rgb_image, step, thin_mask):

    imff = rgb_image.copy()
    h, w = imff.shape[:2]

    paint_x_start = max(0, startx - step // 2)
    paint_y_start = max(0, starty - step // 2)

    paint_x_end = min(startx + step // 2, w)
    paint_y_end = min(starty + step // 2, h)

    #imff = cv2.cvtColor(imff,cv2.COLOR_BGR2HSV)

    threshold = (2, 2, 2)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1,1:-1] = thin_mask
    cv2.floodFill(imff, mask, (startx, starty), newVal=(255, 0, 0), loDiff=threshold, upDiff=threshold,
                  flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)#|cv2.FLOODFILL_FIXED_RANGE)
    mask[1:-1, 1:-1] = mask[1:-1, 1:-1] & ~thin_mask

    return mask[1:-1,1:-1]


def radius_eq_y(a, b, c, lat, long):
    return c / (np.cos(lat) - a*np.sin(lat)*np.sin(PI-long) - b*np.sin(lat)*np.cos(PI-long))


def radius_eq_x(a, b, c, lat, long):
    return c / (np.sin(lat)*np.cos(PI-long) - a*np.cos(lat) - b*np.sin(lat)*np.sin(PI-long))


def radius_eq_z(a, b, c, lat, long):
    return c / (np.sin(lat)*np.sin(PI-long) - a*np.sin(lat)*np.cos(PI-long) - b*np.cos(lat))


def ang_disparity(baseline, radius, lat):
    ad = np.arctan(np.sin(lat)/(baseline/radius + np.cos(lat))) - lat
    ad_PI = ad+PI
    c = np.stack([ad,ad_PI],axis=-1)
    choice=np.argmin(abs(c), axis=-1)
    return np.choose(choice, [ad,ad_PI])


def pt_disparity(ang_disparity, unit_h):
    return (((ang_disparity/(unit_h*PI)) - DISP_OFFSET ) * DISP_SCALE).astype(np.uint8)


def find_planes(pc, rgb_image, edges_image, depth_image, thin_edges, baseline):

    import matplotlib.pyplot as plt

    wx, wy, wz, lat, long, wrd = tuple(range(6))
    h,w = edges_image.shape
    complete_region_mask = np.zeros((h, w), np.uint8)
    inf_region_mask = np.zeros((h, w), np.uint8)
    close_region_mask = np.zeros((h, w), np.uint8)
    new_depth_image = depth_image.copy()

    step=75

    combined = rgb_image.copy()

    for startx in np.arange(0,w, step):
        for starty in np.arange(250, h-250, step):

            if (inf_region_mask[starty,startx] > 0) or \
               (close_region_mask[starty,startx] > 0) or \
               (complete_region_mask[starty,startx] > 0):
                continue


            paint_x_start = max(0, startx - step // 2)
            paint_y_start = max(0, starty - step // 2)

            paint_x_end = min(startx + step // 2, w)
            paint_y_end = min(starty + step // 2, h)

            region_mask = find_region(startx, starty, rgb_image, step, thin_edges)
            edges_mask = region_mask & edges_image





            fpc1 = pc[(edges_mask>0) & ((pc[:,:,wx]!=0)|(pc[:,:,wy]!=0)|(pc[:,:,wz]!=0))]

            planes=[]
            planes.append(plane_estimate(fpc1, wz, wx, wy))  # y = az + bx + c (0)
            planes.append(plane_estimate(fpc1, wx, wy, wz))  # z = ax + by + c (1)
            planes.append(plane_estimate(fpc1, wy, wz, wx))  # x = ay + bz + c (2)
            planes=np.array(planes)
            eq = np.argmin(abs(planes[:,0]) + abs(planes[:,1]))
            a, b, c = planes[eq]

            height, width, __ = rgb_image.shape
            unit_h = 1.0 / height
            unit_w = 2.0 / width

            fpc2 = pc[region_mask>0]
            #fpc = pc[paint_y_start:paint_y_end,paint_x_start:paint_x_end][region_mask[paint_y_start:paint_y_end,paint_x_start:paint_x_end]>0]
            #[paint_y_start:paint_y_end,paint_x_start:paint_x_end]



            #print(eq, a,b,c)
            if (abs(a)< 0.3 and abs(b)<0.3):

                combined[:, :, 0] = rgb_image[:, :, 0] / 2 + thin_edges / 4 + region_mask / 4
                combined[:, :, 1] = rgb_image[:, :, 1] / 2 + complete_region_mask / 2
                combined[:, :, 2] = rgb_image[:, :, 2] / 2 + inf_region_mask / 2
                #cv2.imshow("Work", combined)
                #cv2.waitKey(1)

                if (eq == 0):
                    if  (abs(a)< 0.1 and abs(b)<0.1):
                        a, b, c = 0., 0., np.nanmedian(fpc1[:, wy])
                    rd = radius_eq_y(a, b, c, fpc2[:, lat], fpc2[:, long])

                elif (eq == 1):
                    if  (abs(a)< 0.1 and abs(b)<0.1):
                        a, b, c = 0., 0., np.nanmedian(fpc1[:, wz])
                    rd = radius_eq_z(a, b, c, fpc2[:, lat], fpc2[:, long])

                elif (eq == 2):
                    if  (abs(a)< 0.1 and abs(b)<0.1):
                        a, b, c = 0., 0., np.nanmedian(fpc1[:, wx])
                    rd = radius_eq_x(a, b, c, fpc2[:, lat], fpc2[:, long])

                ad = ang_disparity(baseline, rd, fpc2[:,lat])

                new_disparity = pt_disparity(ad, unit_h)

                if np.max(new_disparity) > 235:
                    #print("x:%4d y:%4d eq:%d GOOD Plane - a:%4.2f b:%4.2f c:%4.2f infinity back projection %d"
                    #      %(startx,starty, eq, a, b, c, np.max(new_disparity)))
                    inf_region_mask = inf_region_mask | region_mask
                elif np.min(new_disparity) < 20:
                    #print("x:%4d y:%4d eq:%d GOOD Plane - a:%4.2f b:%4.2f c:%4.2f too close back projection %d"
                     #     %(startx,starty, eq, a, b, c, np.min(new_disparity)))
                    close_region_mask = close_region_mask | region_mask
                else:
                    new_depth_image[region_mask>0] = new_disparity
                    complete_region_mask = complete_region_mask | region_mask
                    #print("x:%4d y:%4d eq:%d GOOD Plane - a:%4.2f b:%4.2f c:%4.2f good back projection %d-%d"
                    #      %(startx,starty, eq, a, b, c, np.min(new_disparity), np.max(new_disparity)))
                    #cv2.imshow("Output", new_depth_image)
                combined[:, :, 0] = rgb_image[:, :, 0] / 2 + thin_edges / 4
                combined[:, :, 1] = rgb_image[:, :, 1] / 2 + complete_region_mask / 2
                combined[:, :, 2] = rgb_image[:, :, 2] / 2 + inf_region_mask / 2
                plt.imshow(combined)
                plt.draw()
                plt.pause(0.0001)
                #cv2.imshow("Work", combined)
                #cv2.waitKey(1)

    combined[:, :, 0] = rgb_image[:, :, 0] / 2 + thin_edges / 4
    combined[:, :, 1] = rgb_image[:, :, 1] / 2 + complete_region_mask / 2
    combined[:, :, 2] = rgb_image[:, :, 2] / 2 + inf_region_mask / 2
    #cv2.imshow("Work", combined)
    new_depth_image[:250] = 0
    new_depth_image[-250:] = 0
    #cv2.imshow("Output", new_depth_image)
    #cv2.waitKey(1)

    return new_depth_image, complete_region_mask, edges_mask, inf_region_mask, close_region_mask


def get_x(radius, latitude, longitude):
    return radius*np.sin(latitude)*np.cos(PI - longitude)

def get_z(radius, latitude, longitude):
  return radius*np.sin(latitude)*np.sin(PI - longitude)

def get_y(radius, latitude, longitude):
   return  radius*np.cos(latitude)


def fix_limits(pc, depth_image, ceil_height, floor_height, front_dist, back_dist, right_dist, left_dist, baseline):
    
  
    wx, wy, wz, lat, long, wrd = tuple(range(6))

    fpc1 = pc[250:253,:][pc[250:253, :, wy] <1000]
    __, __, ceil_height = plane_estimate(fpc1, wz, wx, wy)  # y = az + bx + c (0)

    fpc1 = pc[-253:-250,:][pc[-253:-250, :, wy] <1000]
    __, __, floor_height = plane_estimate(fpc1, wz, wx, wy)  # y = az + bx + c (0)


    #ceil_height = np.median(pc[248:250,:,wy])
    print("Adjusted ceil_height",ceil_height)
    ##floor_height = np.median(pc[-252:-250,:,wy])
    print("Adjusted floor_height",floor_height)
    new_depth_image = depth_image.copy()
    h,w = depth_image.shape
    unit_h = 1.0 / h
    ##top
    fpc = pc[:250,:][pc[:250, :, wy] == 0]
    rd = radius_eq_y(0, 0, ceil_height, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[:250,:] = pt_disparity(ad, unit_h).reshape(new_depth_image[:250,:].shape)

    fpc = pc[pc[:, :, wy] >  ceil_height]
    rd = radius_eq_y(0, 0, ceil_height, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[pc[:, :, wy] >  ceil_height] = pt_disparity(ad, unit_h)
    #new_depth_image[pc[:, :, wy] >  ceil_height] = 0
    pc[pc[:, :, wy] >  ceil_height, wx] = get_x(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wy] >  ceil_height, wz] = get_z(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wy] >  ceil_height, wy] = get_y(rd, fpc[:, lat], fpc[:, long])

    ##floor
    fpc = pc[-250:,:][pc[-250:, :, wy] == 0]
    rd = radius_eq_y(0, 0, floor_height, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[-250:,:] = pt_disparity(ad, unit_h).reshape(new_depth_image[-250:,:].shape)

    fpc = pc[pc[:, :, wy] <  floor_height]
    rd = radius_eq_y(0, 0, floor_height, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[pc[:, :, wy] <  floor_height] = pt_disparity(ad, unit_h)
    #new_depth_image[pc[:, :, wy] <  floor_height] = 0
    pc[pc[:, :, wy] <  floor_height, wx] = get_x(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wy] <  floor_height, wz] = get_z(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wy] <  floor_height, wy] = get_y(rd, fpc[:, lat], fpc[:, long])

    ##left
    fpc = pc[pc[:, :, wx] <  left_dist]
    rd = radius_eq_x(0, 0, left_dist, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[pc[:, :, wx] <  left_dist] = pt_disparity(ad, unit_h)
    #new_depth_image[pc[:, :, wx] <  left_dist] = 0
    pc[pc[:, :, wx] < left_dist, wy] = get_y(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wx] < left_dist, wz] = get_z(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wx] < left_dist, wx] = get_x(rd, fpc[:, lat], fpc[:, long])

    ##right
    fpc = pc[pc[:, :, wx] >  right_dist]
    rd = radius_eq_x(0, 0, right_dist, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[pc[:, :, wx] >  right_dist] = pt_disparity(ad, unit_h)
    #new_depth_image[pc[:, :, wx] >  right_dist] = 0
    pc[pc[:, :, wx] > right_dist, wy] = get_y(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wx] > right_dist, wz] = get_z(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wx] > right_dist, wx] = get_x(rd, fpc[:, lat], fpc[:, long])

    ##back
    fpc = pc[pc[:, :, wz] <  back_dist]
    rd = radius_eq_z(0, 0, back_dist, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[pc[:, :, wz] <  back_dist] = pt_disparity(ad, unit_h)
    #new_depth_image[pc[:, :, wz] <  back_dist] = 0
    pc[pc[:, :, wz] <  back_dist, wx] = get_x(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wz] <  back_dist, wy] = get_y(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wz] <  back_dist, wz] = get_z(rd, fpc[:, lat], fpc[:, long])


    ##front
    fpc = pc[pc[:, :, wz] > front_dist]
    rd = radius_eq_z(0, 0, front_dist, fpc[:, lat], fpc[:, long])
    ad = ang_disparity(baseline, rd, fpc[:, lat])
    new_depth_image[pc[:, :, wz] > front_dist] = pt_disparity(ad, unit_h)
    #new_depth_image[pc[:, :, wz] > front_dist] = 0
    pc[pc[:, :, wz] > front_dist, wx] = get_x(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wz] > front_dist, wy] = get_y(rd, fpc[:, lat], fpc[:, long])
    pc[pc[:, :, wz] > front_dist, wz] = get_z(rd, fpc[:, lat], fpc[:, long])
    
    return new_depth_image
 


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6



# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


