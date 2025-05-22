import numpy as np
# from tools.nuscenes import NuScenes
import blosc
import cv2
import torch
from bev2PV import CamProjector
from scipy.spatial.transform import Rotation
import json
import os
import glob
import numpy as np

def visualize_mask(mask, img_path=None, alpha=0.5):
    """
    Visualize segmentation mask, optionally overlaying it on an image.
    
    Args:
        mask (np.ndarray): Segmentation mask of shape (H, W, C) where C is number of classes
        img_path (str, optional): Path to background image. If None, shows mask on black background
        alpha (float): Transparency for the overlay (0.0 to 1.0)
    
    Returns:
        np.ndarray: Visualization image in BGR format
    """
    # Define colors for each class (in BGR format)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 0),    # Dark blue
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark red
        (128, 128, 0),  # Olive
    ]
    
    # Create visualization image
    if img_path is not None:
        vis_img = cv2.imread(img_path)
        vis_img = cv2.resize(vis_img, (mask.shape[1], mask.shape[0]))
        if vis_img is None:
            raise ValueError(f"Could not read image at {img_path}")
    else:
        vis_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Create colored overlay
    overlay = np.zeros_like(vis_img, dtype=np.float32)
    
    # Add each class with its color
    for c in range(mask.shape[-1]):
        color = colors[c % len(colors)]
        for i in range(3):  # BGR channels
            overlay[..., i] += mask[..., c] * color[i]
    
    # Normalize overlay to 0-255 range
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Blend overlay with original image
    vis_img = cv2.addWeighted(vis_img, 1.0, overlay, alpha, 0)
    
    return vis_img

def visualize_contours(mask, contours, img_path=None):
    """
    Visualize contours over a mask or image.
    
    Args:
        mask (np.ndarray): Binary mask
        contours (list): List of contours from cv2.findContours
        img_path (str, optional): Path to background image
    
    Returns:
        np.ndarray: Visualization image with contours
    """
    if img_path is not None:
        vis_img = cv2.imread(img_path)
        vis_img = cv2.resize(vis_img, (mask.shape[1], mask.shape[0]))
    else:
        vis_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
    # Draw the mask in red
    vis_img[mask > 0] = [0, 0, 255]
    
    # Draw contours in green
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    return vis_img

# Example usage:
"""
# After getting the perspective mask:
perspective_mask = bev_to_camera_perspective(bev_mask, cam_intrinsic, rotation, translation, (1080, 1920))

# Visualize mask only
vis_img = visualize_mask(perspective_mask)

# Or visualize overlaid on camera image
vis_img = visualize_mask(perspective_mask, img_path=img_path, alpha=0.5)

# Display
cv2.imshow('Visualization', vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Or save
cv2.imwrite('visualization.png', vis_img)
"""

def get_gt(blosc_path, shape=(448, 560)):
    with open(blosc_path, 'rb') as f:
        b = blosc.decompress(f.read())
        blosc_file = np.frombuffer(b, dtype=np.bool_).reshape(*shape, -1)
        
    occ_mask = blosc_file[:,:,17]
    occ_mask = (occ_mask == 0)
    blosc_file = blosc_file[:,:,[0, 1, 2, 3, 4, 5, 6, 8, 9, 12]]
    blosc_file *= np.expand_dims(occ_mask,2)
    return blosc_file

def get_fov_from_intrinsics(camera_intrinsic, img_shape):
    """
    Calculate horizontal and vertical field of view (FOV) from camera intrinsics.
    
    Args:
        camera_intrinsic (np.ndarray): 3x3 camera intrinsic matrix
        img_shape (tuple): Image shape (height, width)
        
    Returns:
        tuple: (horizontal_fov, vertical_fov) in degrees
    """
    # Get focal lengths and principal point from intrinsic matrix
    fx = camera_intrinsic[0, 0]  # focal length in x direction
    fy = camera_intrinsic[1, 1]  # focal length in y direction
    width = img_shape[1]
    height = img_shape[0]
    
    # Calculate FOV using arctangent
    horizontal_fov = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    vertical_fov = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
    
    return horizontal_fov, vertical_fov


# Transform to perspective view
def bev_to_camera_perspective(bev_mask, camera_intrinsic, euler_angles, translation_vector, img_shape,
                            fov, bev_range=[-30, 75, -60, 60]):
    """
    Transform BEV mask to perspective view using CamProjector
    """
    # Get BEV grid dimensions
    # bev_mask = bev_mask[:,:168,:]
    bev_height, bev_width = bev_mask.shape[:2]
    x_min, x_max, y_min, y_max = bev_range
    
    # translation_vector /= 1.625
    # euler_angles = [0.349, 8.539, -0.144]
    cam_projector = CamProjector(cam_pos=translation_vector, cam_heading=euler_angles, cam_fov=fov)
    
    # Initialize perspective mask
    perspective_mask = np.zeros((img_shape[0], img_shape[1], bev_mask.shape[-1]), dtype=np.uint8)
    
    # Process each class
    for c in range(bev_mask.shape[-1]):
        class_mask = bev_mask[..., c].astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Visualize contours for verification
        contour_vis = visualize_contours(class_mask, contours)
        cv2.imwrite(f'contours_class_{c}.png', contour_vis)
        
        class_perspective = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
                
            contour_points = contour.reshape(-1, 2)  # points are in (x,y) order
            if len(contour_points) < 3:
                continue
            
            # Convert contour points to BEV metric coordinates
            x_bev = x_min + (contour_points[:, 0] / bev_width) * (x_max - x_min)  # x uses width
            y_bev = y_min + (contour_points[:, 1] / bev_height) * (y_max - y_min)  # y uses height
            z_bev = np.zeros_like(x_bev)
            ones_bev = np.ones_like(x_bev)
            

            bev_points = np.stack([x_bev, -y_bev, z_bev, ones_bev], axis=1)
            
            # Project points using CamProjector
            projected_points = cam_projector.getProjectedPoints(bev_points, img_width=img_shape[1])
            
            if projected_points is None or projected_points.size == 0:
                continue
                
            # Convert to image coordinates
            image_points = projected_points + np.array([img_shape[1]/2, img_shape[0]/2])
            image_points = image_points.astype(np.int32)
            
            # # Filter valid points
            # valid_mask = (
            #     (image_points[:, 0] >= 0) & 
            #     (image_points[:, 0] < img_shape[1]) &
            #     (image_points[:, 1] >= 0) & 
            #     (image_points[:, 1] < img_shape[0])
            # )
            
            # if not valid_mask.any():
            #     continue
                
            # image_points = image_points[valid_mask]
            
            if len(image_points) >= 3:
                cv2.fillPoly(class_perspective, [image_points], 255)
        
        perspective_mask[..., c] = (class_perspective > 0).astype(np.uint8)
    
    return perspective_mask

def calc_rotation_matrix(yaw, pitch, roll):
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)
    # inputs in rad
    Rx = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])

    # Combine rotation matrices
    R = Rz.dot(Ry).dot(Rx)
    return R

def bev_to_camera_perspective_2(bev_mask, camera_intrinsic, euler_angles, translation_vector, img_shape,
                            fov, bev_range=[-30, 75, -60, 60]):
    """
    Transform BEV mask to perspective view using CamProjector
    """
    h,w = img_shape
    k_inv = np.linalg.inv(camera_intrinsic)
    O_c_w = translation_vector
    R_c_cTag = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
    R_w_cTag = calc_rotation_matrix(euler_angles[0], euler_angles[1], euler_angles[2])
    precalcedMM_dims = {'h': 448, 'w': 560}
    precalcedMM_centerPixel = {'Ox': 280, 'Oy': 224}
    precalcedMM_meter2pixel = 75./280

    T_w_cTag = np.zeros((4,4))
    T_w_cTag[:3,:3] = R_w_cTag
    T_w_cTag[:3,3] = O_c_w
    T_w_cTag[3,3] = 1

    T_cTag_c = np.zeros((4,4))
    T_cTag_c[:3,:3] = R_c_cTag.T
    T_cTag_c[3,3] = 1

    # normal of ground
    hat_n_w = np.array([[0], [0], [1]])

    # normal of ground in camera frame
    hat_n_c = (R_c_cTag @ R_w_cTag.T @ hat_n_w)

    # Create grid of pixel coordinates
    u = np.repeat(np.arange(w)[None,:], h, 0)
    v = np.repeat(np.arange(h)[:,None], w, 1)
    U = np.concatenate((u[None,:,:], v[None,:,:]), axis=0)
    U_i = U.astype(np.float32)
    U_1d = U_i.reshape(2, -1).T

    # homogeneous coordinates in camera frame
    bar_U_1D = np.concatenate((U_1d, np.ones_like(U_1d[:,0])[:,None]), axis=1)[:, :, None]
    
    # ray direction
    d = k_inv @ bar_U_1D
    
    tilde_x_c_1D = np.concatenate((d,-(1 / (hat_n_w.T @ O_c_w)) * hat_n_c.T @ d), axis=1)


    tilde_x_w_1D = T_w_cTag @ T_cTag_c @ tilde_x_c_1D

    x_w_1D = tilde_x_w_1D[:, :3] / np.repeat(tilde_x_w_1D[:, 3:4], 3, 1) 
    x_c_1D = tilde_x_c_1D[:, :3] / np.repeat(tilde_x_c_1D[:, 3:4], 3, 1)

    x_e_1D = x_w_1D.copy()
    # x_e_1D[:,1,0] = -x_e_1D[:,1,0]
    x_e_1D[:, 2, 0] = -x_e_1D[:, 2, 0]
    
    valid_indices_1D = x_c_1D[:, 2, 0] > 0
    valid_indices_2D = valid_indices_1D[:, None].reshape(h, w)
    u, v = x_e_1D[:, 0:1, 0].reshape(h, w), x_e_1D[:, 1:2, 0].reshape(h, w) 
    u = u / precalcedMM_meter2pixel + precalcedMM_centerPixel['Ox'] # pixel
    v = v / precalcedMM_meter2pixel + precalcedMM_centerPixel['Oy'] # pixel
                                                              

    grid = torch.cat((torch.tensor(u[:, :, None] / precalcedMM_dims['w']),torch.tensor(v[:, :, None] / precalcedMM_dims['h'])), dim=-1)[None]
    grid = grid * 2 - 1
    grid[torch.logical_not(torch.tensor(valid_indices_2D)[None, :, :, None].expand(-1, -1, -1, 2))] = 1000
    grid_i = grid.float()
    bev_mask = bev_mask.astype(np.float32)
    bev_mask = torch.tensor(bev_mask)[None,:,:,:].permute(0,3,1,2)
    I_i = torch.nn.functional.grid_sample(bev_mask, grid_i)
    I_i = I_i.numpy().squeeze(0).transpose(1,2,0)
    return I_i
    

# Initialize NuScenes
# version = 'trainval-mini'
# root_path = '/home/sonaalk/imagry/data/converted/'
# base_path = '/home/sonaalk/imagry/data/trips/'
# nusc = NuScenes(version=version, dataroot=root_path, base_path=base_path, verbose=True)


# # Get sample data
# samp = nusc.sample[0]

# These euler angles are take from cam_config.json
cam_to_euler_angles = {
    'FRONT_LEFT': [52.117, 10.134, -2.515],
    'BACK_LEFT': [138.698, -0.235, 0.425],
    'FRONT': [-0.349, -8.539, -0.144],
    'FRONT_MID_RANGE': [-0.349, -1.411, 0.886],
    'FRONT_LONG_RANGE': [-0.212, -1.630, 0.438],
    'BACK_RIGHT': [-142.955, 0.167, 1.852],
    'FRONT_RIGHT': [-51.833, 10.538, 4.044],
    'BACK': [180.583, 3.202, 1.111],
}




trip_path = "/home/sonaalk/imagry/data/trips/trainval-mini/2024-06-13T15_55_46/"
bev_mask_path = "/home/Data/mm/san_jose_entron_train_04_04/2024-06-13T15_55_46/bigmap/trip_0_500_mm_0.blosc"

cam_configs = json.load(open(trip_path + 'cams_configs.json'))

# First, let's create a function to parse the camera matrix string
def parse_camera_matrix(matrix_str):
    """Convert camera matrix string to numpy array"""
    return np.array([float(x) for x in matrix_str.split(',')]).reshape(3, 3)


# Modify the for loop to use camera configurations
for cam_config in cam_configs['cams']:
    # Get camera parameters from config
    translation = np.array([cam_config['x'], -cam_config['y'], -cam_config['z']])
    euler_angles = [-cam_config['heading'], -cam_config['pitch'], cam_config['roll']]
        
    # Parse camera matrix
    cam_intrinsic = parse_camera_matrix(cam_config['new_camera_matrix'])
    
    img_path = sorted(glob.glob(f"{trip_path}/3d_images/{cam_config['index']}/left/*.jpeg"))[0]  # Adjust this path as needed
    print("CAMERA:", cam_config['topic'], "IMAGE PATH:", img_path)
    # Get BEV mask
    bev_mask = get_gt(bev_mask_path)

    # Get FOV from config instead of calculating
    horizontal_fov = cam_config['h_fov']
    vertical_fov = cam_config['v_fov']

    # Transform to perspective view
    pv_mask = bev_to_camera_perspective_2(
        bev_mask, 
        cam_intrinsic, 
        euler_angles, 
        translation, 
        (1080, 1920),
        horizontal_fov
    )

    # Visualize and save
    vis_pv_mask = visualize_mask(pv_mask, img_path)
    cv2.imwrite(f'visualization_{cam_config["topic"]}.png', vis_pv_mask)

vis_bev_mask = visualize_mask(bev_mask)
cv2.imwrite('gt_mask.png', vis_bev_mask)

print("Masks saved as 'bev_mask.npy' and 'pv_mask.npy'") 