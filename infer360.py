
# Based on ideas from: https://gitlab.com/UnBVision/edgenet360/-/blob/master/infer360.py?ref_type=heads

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.py_cuda import lib_mdbnet360_setup, get_point_cloud, get_voxels, downsample_grid, get_ftsdf, downsample_limits
from utils.visual_utils import obj_export
from utils.spher_cubic_proj import spher_cubic_proj
from model.mdbnet import get_res_unet_rgb, get_2Dfeatures, get_activation
from utils.post_process import voxel_filter, voxel_fill, fill_limits_vox, instance_remover,remove_internal_voxels_v2
import cv2
import math

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CV_PI = 3.141592


prediction_shape = (60,36,60)


DATA_PATH = './data'
#OUTPUT_PATH = './output/model_Y'
OUTPUT_PATH = './output'
WEIGHTS_PATH = './weights'
BASELINE = 0.264 # for meeting room and Usability
#BASELINE = 0.176 for Kitchen scene,  0.202 for Studio scene
V_UNIT = 0.02
NETWORK = 'MDBNet'
FILTER = True
SMOOTHING = True
REMOVE_INTERNAL = False
MIN_VOXELS = 15
TRIANGULAR_FACES = False
FILL_LIMITS = True
INNER_FACES = False
INCLUDE_TOP = False
SAVED_WEIGHTES_3D = "./weights/ResUNet_rgb_late_tanh_identity_fold1_2024-01-26.pth"
SAVED_WEIGHTES_2D ="./weights/2D_pretrained_late_tanh_identity_fold1_2024-01-26.pth"
PRE_TRAINED_MODEL = "./pretrained_segformer_b5_ade-640-640"
FUSION_LOC = 'late'
IMAGE_SIZE = (480, 640)
SENSOR_W = 640  # perspective image dim
SENSOR_H = 480  

      
def process(depth_file, rgb_file, out_prefix):
    
    print("Processing point cloud...")
    

   
    wx, wy, wz, lat, long, wrd = tuple(range(6))

    
    point_cloud, depth_image = get_point_cloud(depth_file, baseline=BASELINE)
    print("point_cloud", point_cloud.shape)
   
    ceil_height, floor_height = np.max(point_cloud[:,:,wy]), np.min(point_cloud[:,:,wy])
    front_dist, back_dist = np.max(point_cloud[:,:,wz]), np.min(point_cloud[:,:,wz])
    right_dist, left_dist = np.max(point_cloud[:,:,wx]), np.min(point_cloud[:,:,wx])
    print ('floor_height', floor_height)
    print("room height: %2.2f (%2.2f <> %2.2f)" % (ceil_height - floor_height, ceil_height, floor_height))
    print("room width:  %2.2f (%2.2f <> %2.2f)" % ( right_dist - left_dist, right_dist , left_dist))
    print("room length: %2.2f (%2.2f <> %2.2f)" % ( front_dist - back_dist, front_dist , back_dist))
    
    cam_hight = - floor_height  
    if ((DATASET == 'Kitchen') or (DATASET == 'kitchen') ):
        average = (ceil_height + abs(floor_height)) / 2
        average_str = str(average)
        decimal_pos = average_str.find('.')
        result_str = average_str[:decimal_pos + 2]
        result = float(result_str)
        cam_hight = result
    
    
    cam_back = 0.0 # un-used
    print('CAM height:', cam_hight)
    camx, camy, camz = -left_dist, cam_hight, -back_dist
    
    print('camx', camx)
    print('camy', camy)
    print('camz', camz)
    lib_mdbnet360_setup(device=0, num_threads=1024, v_unit=V_UNIT, v_margin=0.24, f=518.8579, debug=0, cam_h = cam_hight, cam_b = cam_back)
    
    #########################################################################################################
    print("\nLoading %s..." % NETWORK)
    model2D, img_processor = get_2Dfeatures(PRE_TRAINED_MODEL) 
    model = get_res_unet_rgb(FUSION_LOC) # 3D model
    
     # make device agnostic code
    dev = 'cuda:0'
    device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")
    # 1- load the saved state dict
    state_dict_2d = torch.load(f=SAVED_WEIGHTES_2D)
    state_dict_3d = torch.load(f=SAVED_WEIGHTES_3D)
    
    # 2- create a new state dict in which 'module.' prefix is removed 'if multple GPUs'
    new_state_dict_2d = {k.replace('module.', ''): v for k, v in state_dict_2d.items()}
    new_state_dict_3d = {k.replace('module.', ''): v for k, v in state_dict_3d.items()} 
    
    # 3- load the new state dict to your model      
    model2D.load_state_dict(new_state_dict_2d)
    model.load_state_dict(new_state_dict_3d)
    
    # send model to GPU
    model2D = model2D.to(device)
    model = model.to(device)
    
    ###########################################################################################################################################
    
    xs, ys, zs = prediction_shape
    
    pred_full = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
    flags_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
    surf_full = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
   
    ### spherical to cubic projection
    rgb_views,depth_views = spher_cubic_proj(depth_file, rgb_file, out_prefix) # return lists
   
    print("Inferring...")
    
    for (rgb_view_idx, rgb_filename) in (rgb_views):
        print(f"RGB view: {rgb_view_idx}, RGB Filename: {rgb_filename}")
        
        view = rgb_view_idx 
        
        rgb = cv2.imread(rgb_filename, cv2.IMREAD_COLOR)
        
        vox_grid, depth_mapping_3d = get_voxels(depth_image, point_cloud, depth_image.shape,
                              min_x=left_dist , max_x=right_dist ,
                              min_y=floor_height , max_y=ceil_height ,
                              min_z=back_dist , max_z=front_dist ,
                              vol_number=view)
        
        vox_tsdf, vox_limits = get_ftsdf(depth_image, vox_grid,
                                         min_x=left_dist , max_x=right_dist ,
                                         min_y=floor_height , max_y=ceil_height ,
                                         min_z=back_dist , max_z=front_dist , baseline=BASELINE, vol_number=view)
        
        vox_grid_down = downsample_grid(vox_grid)  
        
        #################################################################
        print("Shape of depth_mapping:", depth_mapping_3d.shape)
        depth_m = depth_mapping_3d.flatten()
        valid_depth_values = np.where(depth_m>= 0)
        print('valid_depth_values len', len(valid_depth_values))
        print('valid_depth_values shape', valid_depth_values)
        print('depth_mapping[valid_depth_values].max',depth_m[valid_depth_values].max())
        print('depth_mapping[valid_depth_values].min',depth_m[valid_depth_values].min())
        
        ##############################################################
        depth_mapping = depth_mapping_3d.reshape(1,240*144*240)
        rgb_v = cv2.imread(rgb_filename, cv2.IMREAD_COLOR)
        # convert to tensor
        rgb_t = torch.from_numpy(rgb_v.astype(np.float32).transpose((2, 0, 1)) / 255)
        rgb_t = rgb_t.to(device)
        
        x= torch.from_numpy(vox_tsdf).reshape(1,1, 240, 144, 240).to(device)
        # get the predictions
        model.eval()
        model2D.eval()
        with torch.inference_mode():
        
          # get the 2d feature maps from pretrained model
          rgb_t = rgb_t.unsqueeze(0)
          imgs = img_processor(images=rgb_t, return_tensors="pt").to(device)
          
          hook,activation = get_activation('linear_fuse')
          model2D.decode_head.register_forward_hook(hook)
          cls = model2D(**imgs) # predected 2d classes
          classes_logits = cls.logits # logits from classification layer
          feature2d = activation['linear_fuse'] # the activation maps befor the classification layer
          
          # upsample the output to match input images size
          desired_size = IMAGE_SIZE
          feature2d = F.interpolate(feature2d, size=desired_size, mode='bilinear', align_corners=True)
          
          # input the predicted 2D features to the 3D model alongside with depth data
          
          depth_3d = torch.from_numpy(depth_mapping).to(device)
          print("Shape of rgb :", rgb_t.shape)
          print("Shape of F-tsdf :", x.shape)
          print("Shape of depth_3d:", depth_3d.shape)
          
          pred = model(rgb_t, x,feature2d, depth_3d, device).cpu()
          
          print("Shape of pred torch:", pred.shape)
          # reshape the pred tensor to [60, 36, 60, 12]
          pred_reshaped = pred.permute(2, 3, 4, 1, 0).contiguous().view(zs,ys,xs, 12)
          pred_reshaped = pred_reshaped.numpy()
          print("Shape of pred_reshaped numpy:", pred_reshaped.shape)
        ####################################################################################################
          flags_down = downsample_limits(vox_limits)
          
          print("Shape of flags_down:", flags_down.shape)
          
          # repeat the flags_down array along the last dimension
          flags_down_repeated = np.repeat(flags_down[:, :, :, np.newaxis], 12, axis=3)
          print("Shape of flags_down_repeated:", flags_down_repeated.shape)
          fpred = pred_reshaped * flags_down_repeated

          print("Shape of fpred:", fpred.shape)
          print("type of fpred:", fpred.dtype)
          
          ##################################################################### to visualise single view#######################################################
          
          if view==1:
              pred_full_ = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
              flags_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              surf_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              
              pred_full_[ zs:, :, xs//2:-xs//2] = fpred
              surf_full_[ zs:, :, xs//2:-xs//2] = vox_grid_down
              
              
  
          elif view==2:
              pred_full_ = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
              flags_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              surf_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              
              pred_full_[zs//2:-zs//2, :, xs:] = np.flip(np.swapaxes(fpred,0,2),axis=0)
              surf_full_[zs//2:-zs//2, :, xs:] = np.flip(np.swapaxes(vox_grid_down,0,2),axis=0)
              
              
          
          elif view==3:
              pred_full_ = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
              flags_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              surf_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              
              pred_full_[:zs, :, xs//2:-xs//2] = np.flip(fpred,axis=[0,2])
              surf_full_[:zs, :, xs//2:-xs//2] = np.flip(vox_grid_down,axis=[0,2])
              
          
          elif view == 4:
              pred_full_ = np.zeros((xs*2,ys,xs*2,12), dtype=np.float32)
              flags_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
              surf_full_ = np.zeros((xs*2,ys,xs*2), dtype=np.uint8)
             
              pred_full_[zs//2:-zs//2, :, :xs] = np.flip(np.swapaxes(fpred,0,2),axis=2)
              surf_full_[zs//2:-zs//2, :, :xs] = np.flip(np.swapaxes(vox_grid_down,0,2),axis=2) 
             
          
          
          y_pred_ = np.argmax(pred_full_, axis=-1)
          # fill camera position
          y_pred_[zs-4:zs+4,0,xs-4:xs+4] = 2
          
          if FILTER:
              print("Filtering...")
              y_pred_ = voxel_filter(y_pred_)
          
          if MIN_VOXELS>1:
              print("Removing small instances (<%d voxels)..." % MIN_VOXELS)
              y_pred_ = instance_remover(y_pred_, min_size=MIN_VOXELS)
          
          if SMOOTHING:
              print("Smoothing...")
              y_pred_ = voxel_fill(y_pred_)
          
          if REMOVE_INTERNAL:
              print("Removing internal voxels of the objects...")
              y_pred_ = remove_internal_voxels_v2(y_pred_, camx, camy, camz, V_UNIT)
         
          print("           ")
          
          out_file = out_prefix + '_surface_'+str(view)
          print("Exporting surface to       %s.obj" % out_file)
          obj_export(out_file, surf_full_, surf_full_.shape, camx, camy, camz, V_UNIT, include_top=INCLUDE_TOP,
                                                                                     triangular=TRIANGULAR_FACES)
          
          out_file = out_prefix+'_prediction_'+str(view)
          print("Exporting prediction to    %s.obj" % out_file)
          obj_export(out_file, y_pred_, (xs*2,ys,zs*2), camx, camy, camz, V_UNIT, include_top=INCLUDE_TOP,
                                                                                 triangular=TRIANGULAR_FACES,
                                                                                 inner_faces=INNER_FACES)
           
    ######################################################################################################################################
    
    ############################################################################### Full Prediction ######################################
  
            
          if view==1:
              pred_full[ zs:, :, xs//2:-xs//2] += fpred
              surf_full[ zs:, :, xs//2:-xs//2] |= vox_grid_down
              
          elif view==2:
              pred_full[zs//2:-zs//2, :, xs:] += np.flip(np.swapaxes(fpred,0,2),axis=0) # addition proba then we take the argmax y_pred
              surf_full[zs//2:-zs//2, :, xs:] |= np.flip(np.swapaxes(vox_grid_down,0,2),axis=0)
              
          elif view==3:
              pred_full[:zs, :, xs//2:-xs//2] += np.flip(fpred,axis=[0,2])
              surf_full[:zs, :, xs//2:-xs//2] |= np.flip(vox_grid_down,axis=[0,2])
          
          elif view == 4: 
              pred_full[zs//2:-zs//2, :, :xs] += np.flip(np.swapaxes(fpred,0,2),axis=2)
              surf_full[zs//2:-zs//2, :, :xs] |= np.flip(np.swapaxes(vox_grid_down,0,2),axis=2)   
    
    print("Combining all views...")
    y_pred = np.argmax(pred_full, axis=-1)
    
    # fill camera position
    y_pred[zs-4:zs+4,0,xs-4:xs+4] = 2
    
    # remove ceiling and floor from the prediction and set them in post-processing with find limits
    y_pred[y_pred == 1] = 0 
    y_pred[y_pred == 2] = 0 
    
    ################################################################################
    
    if FILTER:
        print("Filtering...")
        y_pred = voxel_filter(y_pred)
   
    if MIN_VOXELS>1:
        print("Removing small instances (<%d voxels)..." % MIN_VOXELS)
        y_pred = instance_remover(y_pred, min_size=MIN_VOXELS)
    
    if SMOOTHING:
        print("Smoothing...")
        y_pred = voxel_fill(y_pred)
    
    if FILL_LIMITS:
        print("Completing room limits...")
        print('limits max x,min x,max z,min z:', int(right_dist/V_UNIT), int(left_dist/V_UNIT),int(front_dist/V_UNIT), int(back_dist/V_UNIT))
        y_pred = fill_limits_vox(y_pred,int(right_dist/V_UNIT), int(left_dist/V_UNIT),int(front_dist/V_UNIT), int(back_dist/V_UNIT))
    
    if REMOVE_INTERNAL:
        print("Removing internal voxels of the objects...")
        y_pred = remove_internal_voxels_v2(y_pred, camx, camy, camz, V_UNIT)
    
    print("           ")
    
    out_file = out_prefix + '_surface'
    print("Exporting surface to       %s.obj" % out_file)
    obj_export(out_file, surf_full, surf_full.shape, camx, camy, camz, V_UNIT, include_top=INCLUDE_TOP,
                                                                               triangular=TRIANGULAR_FACES)

    out_file = out_prefix+'_prediction'
    print("Exporting prediction to    %s.obj" % out_file)
    obj_export(out_file, y_pred, (xs*2,ys,zs*2), camx, camy, camz, V_UNIT, include_top=True,
                                                                           triangular=TRIANGULAR_FACES,
                                                                           inner_faces=INNER_FACES)
    
    print("Finished!\n")
    
def parse_arguments():
    global DATA_PATH, DATASET, OUTPUT_PATH, BASELINE, V_UNIT, NETWORK, FILTER, SMOOTHING, \
           FILL_LIMITS, MIN_VOXELS, TRIANGULAR_FACES, WEIGHTS_PATH, INCLUDE_TOP, REMOVE_INTERNAL, INNER_FACES

    print("\nSemantic Scene Completion Inference from 360 depth maps\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",         help="360 dataset dir", type=str)
    parser.add_argument("depth_map",       help="360 depth map", type=str)
    parser.add_argument("rgb_file",        help="rgb", type=str)
    parser.add_argument("output",          help="output file prefix", type=str)
    parser.add_argument("--baseline",      help="Stereo 360 camera baseline. Default %5.3f"%BASELINE, type=float,
                                           default=BASELINE, required=False)
    parser.add_argument("--v_unit",        help="Voxel size. Default %5.3f" % V_UNIT, type=float,
                                           default=V_UNIT, required=False)
    parser.add_argument("--network",       help="Network to be used. Default %s" % NETWORK, type=str,
                                           default=NETWORK, choices=["EdgeNet", "USSCNet"], required=False)
    parser.add_argument("--filter",        help="Apply 3D low-pass filter? Default yes.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--smoothing",     help="Apply smoothing (fill small holes)? Default yes.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--fill_limits",   help="Fill walls on room limits? Default yes.", type=str,
                                           default="Y", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--remove_internal",   help="Remove internal voxels? Default no.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--inner_faces",   help="Include inner faces of objects? Default no.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--min_voxels",    help="Minimum number of voxels per object instance. Default %d."%MIN_VOXELS, type=int,
                                           default=MIN_VOXELS, required=False)
    parser.add_argument("--triangular",    help="Use triangular faces? Default No.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--include_top",   help="Include top (ceiling) in output model? Default No.", type=str,
                                           default="N", choices=["Y", "y", "N", "n"], required=False)
    parser.add_argument("--data_path",     help="Data path. Default %s"%DATA_PATH, type=str,
                                           default=DATA_PATH, required=False)
    parser.add_argument("--output_path",   help="Output path. Default %s"%OUTPUT_PATH, type=str,
                                           default=OUTPUT_PATH, required=False)
    parser.add_argument("--weights_path",   help="Weights path. Default %s"%WEIGHTS_PATH, type=str,
                                           default=WEIGHTS_PATH, required=False)

    args = parser.parse_args()

    BASELINE = args.baseline
    V_UNIT = args.v_unit
    NETWORK = args.network
    FILTER = args.filter in ["Y", "y"]
    SMOOTHING = args.smoothing in ["Y", "y"]
    REMOVE_INTERNAL = args.remove_internal in ["Y", "y"]
    FILL_LIMITS = args.fill_limits in ["Y", "y"]
    INNER_FACES = args.inner_faces in ["Y", "y"]
    MIN_VOXELS = args.min_voxels
    TRIANGULAR_FACES = args.triangular in ["Y", "y"]
    INCLUDE_TOP = args.include_top in ["Y", "y"]
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    WEIGHTS_PATH = args.weights_path

    DATASET = args.dataset
    depth_map = os.path.join(DATA_PATH, DATASET, args.depth_map)
    rgb_file = os.path.join(DATA_PATH, DATASET, args.rgb_file)
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
    print("Baseline:     ", BASELINE)
    print("V_Unit:       ", V_UNIT)
    print("")

    return depth_map, rgb_file, output

# Main Function
def Run():
    depth_map, rgb_file, output = parse_arguments()
    process(depth_map, rgb_file, output)


if __name__ == '__main__':
  Run()
