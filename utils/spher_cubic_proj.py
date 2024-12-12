import argparse
import os
import numpy as np
import cv2
import math

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


CV_PI = 3.141592
#DATA_PATH = './data'
#OUTPUT_PATH = './output'
SENSOR_W = 640  # perspective image dim
SENSOR_H = 480  

def spher_cubic_proj(depth_file, rgb_file, out_prefix):
    
    spher_rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    spher_depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    print('spher_rgb shape', spher_rgb.shape)
    print('spher_depth shape', spher_depth.shape)
    height, width, channels = spher_rgb.shape
   
    
    ########################################### Generate 4 views projection using sphere to cubic projection ##############################
    rgb_views = []
    depth_views = []
    side_range = (SENSOR_W / float(SENSOR_H))	#over projection range for output width
    img_out = np.zeros((SENSOR_H,SENSOR_W,3),np.uint8)
    depth_out = np.zeros((SENSOR_H,SENSOR_W),np.uint8) 
    
        
    #Initial shift as done in the original projection
    init_shift = 0;
    img = np.zeros((height,width,3),np.uint8)
    depth_img = np.zeros((height,width),np.uint8)
    
    if init_shift < 0:
      offset = -1*init_shift
      img[0:height,0:width-offset] = spher_rgb[0:height,offset:width]
      img[0:height,width-offset:width] = spher_rgb[0:height,0:offset]
      ## depth map 
      depth_img[0:height,0:width-offset] = spher_depth[0:height,offset:width]
      depth_img[0:height,width-offset:width] = spher_depth[0:height,0:offset]

    else:
      offset = init_shift
      img[0:height,0:offset] = spher_rgb[0:height,width-offset:width]
      img[0:height,offset:width] = spher_rgb[0:height,0:width-offset]
      ## depth map
      depth_img[0:height,0:offset] = spher_depth[0:height,width-offset:width]
      depth_img[0:height,offset:width] = spher_depth[0:height,0:width-offset]
     
    PI = 3.141592
    
    #For front image
    x=1.0
    for i in range(0, SENSOR_H):
        z=(float(i)/SENSOR_H * (-2.0)) + 1.0
      	
        for j in range(0, SENSOR_W):
            
            y=(float(j)/SENSOR_W * (-2.0 * side_range)) + side_range
            r = np.sqrt(x*x+y*y+z*z)
            theta = np.arccos(z/r)
            
            if x==0:
                phi= 0.5*PI
            else:
                phi = np.arctan(y/x)
            
            a = int(theta / PI * height + 0.5)
            b = width -1 - int(phi / (2*PI) * width + 0.5)
            
            if b < 0:
                b = b+width
            if b > (width-1):
                b = b-width
            
            img_out[i,j] = img[a,b]
            depth_out[i,j] = depth_img[a,b]
          
    # rgb perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Cubic_1.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((1, out_filename))
    # depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Depth_Cubic_1.png'
    cv2.imwrite(out_depth_filename, depth_out)
    depth_views.append((1, out_depth_filename))
    ############################################################################################
    
    #For left image
    y = -1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            x = (float(j) / SENSOR_W * (-2.0 * side_range)) + side_range
            # x = (float(j)/SENSOR_W * (-2.0 )) + 1.0
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            if phi > 0:
                phi = phi + PI
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Cubic_2.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((2, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Depth_Cubic_2.png'
    cv2.imwrite(out_depth_filename, depth_out)  
    depth_views.append((2, out_depth_filename))
    #############################################################################################
    
    #For back image
    x = -1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            y = (float(j) / SENSOR_W * (2.0 * side_range)) - side_range
            # y = (float(j)/SENSOR_W * (2.0)) + 1.0
    
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            phi = phi + PI
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Cubic_3.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((3, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Depth_Cubic_3.png'
    cv2.imwrite(out_depth_filename, depth_out)  
    depth_views.append((3, out_depth_filename))
    ##############################################################################################
    
    
    #For right image
    y= 1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            x = (float(j) / SENSOR_W * (2.0 * side_range)) - side_range
            # x = (float(j)/SENSOR_W * (2.0)) + 1.0
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            if phi < 0:
                phi = phi + PI
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Cubic_4.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((4, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Depth_Cubic_4.png'
    cv2.imwrite(out_depth_filename, depth_out)  
    depth_views.append((4, out_depth_filename))
    
    ###############################################################################################
    '''
    #For top image
    z= 1.0
    for i in range(0, SENSOR_H):
        x=(float(i)/SENSOR_H * (-2.0)) + 1.0
        
        for j in range(0, SENSOR_W):
            y=(float(j)/SENSOR_W * (-2.0 * side_range)) + side_range
            
            r = np.sqrt(x*x+y*y+z*z)
            theta = np.arccos(z/r)
            if x==0:
                phi= 0.5*PI
            else:
                phi = np.arctan(y/x)
            if i>= face_height/2: 
                phi=phi+PI
            
            a = int(theta / PI * height + 0.5)
            b = width -1 - int(phi / (2*PI) * width + 0.5)
            
            if b < 0:
                b = b+width
            if b > (width-1):
                b = b-width
            if a<0:
                a=0
            if a > height-1:
                a=height-1
            
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Cubic_5.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((5, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Depth_Cubic_5.png'
    cv2.imwrite(out_depth_filename, depth_out)  
    depth_views.append((5, out_depth_filename))
    #########################################################################
    
    #For bottom image
    z= -1.0
    for i in range(0, SENSOR_H):
        x=(float(i)/SENSOR_H * (2.0)) - 1.0
        
        for j in range(0, SENSOR_W):
            y=(float(j)/SENSOR_W * (2.0 * side_range)) - side_range
            
            r = np.sqrt(x*x+y*y+z*z)
            theta = np.arccos(z/r)
    
            if x==0:
                phi= 0.5*PI
            else:
                phi = np.arctan(y/x)
            if i<= face_height/2: 
                phi=phi+PI
            
            a = int(theta / PI * height + 0.5)
            b = width -1 - int(phi / (2*PI) * width + 0.5)
            
            if b < 0:
                b = b+width
            if b > (width-1):
                b = b-width
    
            if a<0:
                a=0
            if a > height-1:
                a=height-1
            
    
    o# Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Cubic_6.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((6, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_Depth_Cubic_6.png'
    cv2.imwrite(out_depth_filename, depth_out)  
    depth_views.append((6, out_depth_filename))
'''
##############################################################################################
    
    return rgb_views , depth_views

    '''
    ############################################################  shift 45 to get the rest of the views #########################################################
    rotation = width//8 
    #init_shift = int((-1)*width//4 - rotation)
    init_shift = int(width//4 - rotation)
    PI = 3.141592
    img = np.zeros((height,width,3),np.uint8)
    depth_img = np.zeros((height,width),np.uint8)
    
    if init_shift < 0:
        offset = -1*init_shift
        img[0:height,0:width-offset]  = spher_rgb[0:height,offset:width]
        img[0:height,width-offset:width]= spher_rgb[0:height,0:offset]
        ## depth map
        depth_img[0:height,0:width-offset]  = spher_depth[0:height,offset:width]
        depth_img[0:height,width-offset:width]= spher_depth[0:height,0:offset]
        

    else:
        offset = init_shift
        img[0:height,0:offset] = spher_rgb[0:height,width-offset:width]
        img[0:height,offset:width] = spher_rgb[0:height,0:width-offset]
        ## depth map
        depth_img[0:height,0:offset] = spher_depth[0:height,width-offset:width]
        depth_img[0:height,offset:width] = spher_depth[0:height,0:width-offset]
     
    #For front image
    x = 1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            y = (float(j) / SENSOR_W * (-2.0 * side_range)) + side_range
            # y = (float(j)/SENSOR_W * (-2.0)) + 1.0
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Cubic_1.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((1, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Depth_Cubic_1.png'
    cv2.imwrite(out_depth_filename, depth_out)
    depth_views.append((1, out_depth_filename))

    # For left image
    y = -1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            x = (float(j) / SENSOR_W * (-2.0 * side_range)) + side_range
            # x = (float(j)/SENSOR_W * (-2.0)) + 1.0
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            if phi > 0:
                phi = phi + PI
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Cubic_2.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((2, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Depth_Cubic_2.png'
    cv2.imwrite(out_depth_filename, depth_out)
    depth_views.append((2, out_depth_filename))
    
    # For back image
    x = -1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            y = (float(j) / SENSOR_W * (2.0 * side_range)) - side_range
            # y = (float(j)/SENSOR_W * (2.0)) + 1.0
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            phi = phi + PI
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Cubic_3.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((3, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Depth_Cubic_3.png'
    cv2.imwrite(out_depth_filename, depth_out)
    depth_views.append((3, out_depth_filename))
    
    # For right image
    y = 1.0
    for i in range(0, SENSOR_H):
        z = (float(i) / SENSOR_H * (-2.0)) + 1.0
    
        for j in range(0, SENSOR_W):
            x = (float(j) / SENSOR_W * (2.0 * side_range)) - side_range
            # x = (float(j)/SENSOR_W * (2.0)) + 1.0
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(z / r)
            if x == 0:
                phi = 0.5 * PI
            else:
                phi = np.arctan(y / x)
    
            if phi < 0:
                phi = phi + PI
    
            a = int(theta / PI * height + 0.5)
            b = width - 1 - int(phi / (2 * PI) * width + 0.5)
    
            if b < 0:
                b = b + width
            if b > (width - 1):
                b = b - width
    
            img_out[i, j] = img[a, b]
            depth_out[i, j] = depth_img[a, b]
    
    # Save RGB perspective
    out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Cubic_4.png'
    cv2.imwrite(out_filename, img_out)
    rgb_views.append((4, out_filename))
    
    # Save depth perspective
    out_depth_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}_shift_Depth_Cubic_4.png'
    cv2.imwrite(out_depth_filename, depth_out)
    depth_views.append((4, out_depth_filename))
    ###############################################################################################
    return rgb_views, depth_views
    '''
    
'''    
def parse_arguments():
    global DATA_PATH, OUTPUT_PATH

    print("\nSemantic Scene Completion Inference from 360 depth maps\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",         help="360 dataset dir", type=str)
    parser.add_argument("depth_map",       help="360 depth map", type=str)
    parser.add_argument("rgb_file",        help="rgb", type=str)
    parser.add_argument("output",          help="output file prefix", type=str)
    

    args = parser.parse_args()

    
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    
    dataset = args.dataset
    depth_map = os.path.join(DATA_PATH, dataset, args.depth_map)
    rgb_file = os.path.join(DATA_PATH, dataset, args.rgb_file)
    output = os.path.join(OUTPUT_PATH, args.output)

    fail = False
    if not os.path.isfile(depth_map):
        print("Depth map file not found:", depth_map)
        fail = True

    if not os.path.isfile(rgb_file):
        print("RGB file not found:", rgb_file )
        fail = True

    if fail:
        print("Exiting...\n")
        exit(0)

    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("360 depth map:", depth_map)
    print("360 rgb:      ", rgb_file)
    print("Output prefix:", output)
    print("")

    return depth_map, rgb_file, output, rgb_views, depth_views

# Main Function
def Run():
    depth_map, rgb_file, output = parse_arguments()
    dual_cubic_proj(depth_map, rgb_file, output)


if __name__ == '__main__':
  Run()
'''
'''
#################################################################################################################################################################
import sys
from PIL import Image
from math import pi, sin, cos, tan, atan2, hypot, floor
from numpy import clip

# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# faceIdx is face number
# faceSize is edge length
def outImgToXYZ(i, j, faceIdx):
    #a = 2.0 * float(i) / faceSize
    #b = 2.0 * float(j) / faceSize
    a = 2.0 * float(i) / SENSOR_W
    b = 2.0 * float(j) / SENSOR_H
 
    if faceIdx == 0: # back
        (x,y,z) = (-1.0, 1.0 - a, 1.0 - b)
    elif faceIdx == 1: # left
        (x,y,z) = (a - 1.0, -1.0, 1.0 - b)
    elif faceIdx == 2: # front
        (x,y,z) = (1.0, a - 1.0, 1.0 - b)
    elif faceIdx == 3: # right
        (x,y,z) = (1.0 - a, 1.0, 1.0 - b)
    elif faceIdx == 4: # top
        (x,y,z) = (b - 1.0, a - 1.0, 1.0)
    elif faceIdx == 5: # bottom
        (x,y,z) = (1.0 - b, a - 1.0, -1.0)

    return (x, y, z)

# convert using an inverse transformation
def convertFace(imgIn, imgOut, faceIdx):
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    faceSize = outSize[0]

    for xOut in range(SENSOR_W):
        for yOut in range(SENSOR_H):
            (x,y,z) = outImgToXYZ(xOut, yOut, faceIdx)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2

            # source img coords
            uf = 0.5 * inSize[0] * (theta + pi) / pi
            vf = 0.5 * inSize[0] * (pi/2 - phi) / pi

            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi

            # Pixel values of four corners
            A = inPix[ui % inSize[0], clip(vi, 0, inSize[1]-1)]
            B = inPix[u2 % inSize[0], clip(vi, 0, inSize[1]-1)]
            C = inPix[ui % inSize[0], clip(v2, 0, inSize[1]-1)]
            D = inPix[u2 % inSize[0], clip(v2, 0, inSize[1]-1)]

            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[xOut, yOut] = (int(round(r)), int(round(g)), int(round(b)))


def spher_cubic_proj(depth_file, rgb_file, out_prefix):
    
    spher_rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    spher_depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    print('spher_rgb shape', spher_rgb.shape)
    print('spher_depth shape', spher_depth.shape)
    height, width, channels = spher_rgb.shape
    rgb_views = []
    depth_views = []
    imgIn = Image.open(rgb_file)
    inSize = imgIn.size
    print (inSize)
    faceSize = inSize[0] // 4
    #components = rgb_file.rsplit('.', 2)
    
    FACE_NAMES = {
      #0: 'back', #cubic3
      #1: 'left', # cubic2
      #2: 'front', # cubic1
      #3: 'right', # cubic4
      #4: 'top',
      #5: 'bottom'
      
      0: 'Cubic_1_front', # front
      1: 'Cubic_2_left', # left
      2: 'Cubic_3_right', # back
      3: 'Cubic_4_back', # right
      4: 'top',
      5: 'bottom'
    }
    
    if not os.path.exists('./splitted_spher'):
        os.makedirs('./splitted_spher')
    
    for face in range(6):
        imgOut = Image.new("RGB", (SENSOR_W, SENSOR_H), "black")
        convertFace(imgIn, imgOut, face) # depth_file, rgb_file, out_prefix
        out_filename = f'./splitted_spher/{out_prefix.rsplit("/", 1)[-1]}'
        imgOut.save(out_filename+"_" + FACE_NAMES[face] + ".png")
        rgb_views.append((face+1, out_filename))
        
    return rgb_views, depth_views
'''
    