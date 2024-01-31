# -*- coding: utf-8 -*-
"""
discspeedcalc - a simple tool for calculating translational and rotational
speed of a slow motion video of a disc using ArUco markers.

See example on how to use the script at the bottom of the file.
An example video is provided in the repository, change the path to the actual 
path of the file on your system. This video was taken using a GoPro Hero 10 
camera with a frame rate of 240 FPS and a shutter speed of 1/3840 s. The
disc was illuminated from below using two LED panels of 1100 lux.
The marker type and length needs to be set as parameters. Typically, the
AprilTag 36h11, scaled to a size of 10 cm, have been found to work well.

@Author: Knut Erik T. Giljarhus <knut.erik.giljarhus@gmail.com>
@Links: https://github.com/kegiljarhus/disctracker
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import math

TAG_TYPE = cv2.aruco.DICT_APRILTAG_36h11
TAG_LENGTH = 0.1
SCALE_FACTOR_LIMIT = 0.05
SKEW_LIMIT = 0.55
# STDDEV_LIMIT = 0.004


class Frame:
    def __init__(self):
        pass
    
def debug_store_frame(idx, frame, corners, output_folder):
    color = (0, 255, 0)
    markerType = cv2.MARKER_CROSS
    markerSize = 15
    thickness = 2
    for c in corners[0][0]:
        x,y = [int(a) for a in c]
        cv2.drawMarker(frame, (x, y), color, markerType, markerSize, thickness)
            
    cv2.imwrite(f'{output_folder}\\{idx}.png', frame)

def angle_between(c, p0, p1):
    vec0 = p0 - c
    vec1 = p1 - c
    angle = math.acos(np.dot(vec0,vec1)/(np.linalg.norm(vec0)*np.linalg.norm(vec1)))
    
    return angle

def analyze_image(frame_index, detector, image):
    
    lengths = np.zeros(4)
    angles = np.zeros(4)
    disc_id = None
    
    corners, ids, rejected = detector.detectMarkers(image)       
    
    if np.any(ids != None):
        f = Frame()
        f.disc_id = ids[0][0]
        f.index = frame_index
        
        center = np.mean(corners[0][0], axis=0)
        f.position = center
        p0, p1, p2, p3 = corners[0][0]
        
        lengths[0] = np.linalg.norm(p0 - p1)
        lengths[1] = np.linalg.norm(p1 - p2)
        lengths[2] = np.linalg.norm(p2 - p3)
        lengths[3] = np.linalg.norm(p3 - p0)
        
        f.tag_pixel_length = np.mean(lengths)
        f.scale_factor = TAG_LENGTH / f.tag_pixel_length  # m per pixel
        f.tag_stddev = np.std(f.scale_factor*lengths)
       
        angles[0] = angle_between(center, p0, p1)
        angles[1] = angle_between(center, p1, p2)
        angles[2] = angle_between(center, p2, p3)
        angles[3] = angle_between(center, p3, p0)
        
        f.skewness = max( (np.max(angles) - np.pi)/np.pi, (np.pi - np.min(angles))/np.pi )
        f.angle = np.arctan2(center[1] - p0[1], center[0] - p0[0])
        
        return f
    else:
        return None

def analyze_image_frames(image_paths, time_per_frame):
    """
    Check for markers on a list of images instead. Note that the 
    the video analyzer has an option to export images with detected
    discs. These can then be used here to speed up rechecks.
    """
    # TODO - refactor to common class?
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_TYPE)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    frames = []
    for i, path in enumerate(image_paths):
        image = cv2.imread(path)
        frame = analyze_image(i, detector, image)
        
        if frame is not None:
            frames.append(frame)
    
    if frames:
        # Maybe we should check that all ID's are the same
        disc_id = frames[0].disc_id
    else:
        disc_id = None
        
    return disc_id, frames

def analyze_frames(video_path, output_path=None):
    
    cap = cv2.VideoCapture(video_path)

    
    if not cap.isOpened():
        print("Error opening video file", video_path)
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1.0/fps
    
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_TYPE)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    frame_index = 0
    output_index = 0
    frames = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        
        frame = analyze_image(frame_index, detector, image)
        
        if frame is not None:
            #print('Detected',frame_index)
            
            if output_path:
                cv2.imwrite(f'{output_path}_{output_index}.png', image)
                output_index += 1
            frames.append(frame)
    
        frame_index += 1
    
    if frames:
        disc_id = frames[0].disc_id
    else:
        disc_id = None
        
    return disc_id, frames, time_per_frame

def calculate_speeds(frames, time_per_frame, nmin=0):
    # nmin - starting frame, sometimes frames of holding the disc is caught on camera
    #        resulting in low speeds. These can be eliminated from the speed calculation
    #        by setting an initial value for the iteration
    n = len(frames)
    
    skews = []
    candidates = []
    rejected = []
    for i in range(nmin,n):
        scale_factor_change = abs(frames[i].scale_factor - frames[i-1].scale_factor)/frames[i].scale_factor
        
        distance_pixels = np.linalg.norm(frames[i].position - frames[i-1].position)
        distance = distance_pixels*frames[i].scale_factor
        speed = distance / time_per_frame
        
        angle_change = np.abs(frames[i].angle - frames[i-1].angle)
        # Ensure the angle change is in the range [-pi, pi]
        angle_change = np.abs((angle_change + np.pi) % (2*np.pi) - np.pi)
        
        rotational_speed = angle_change / time_per_frame
        
        if scale_factor_change < SCALE_FACTOR_LIMIT \
            and frames[i].skewness < SKEW_LIMIT and frames[i-1].skewness < SKEW_LIMIT:
          
            # print('accuracy',i,scale_factor_change,skews[-1])
            skews.append(0.5*(frames[i].skewness + frames[i-1].skewness))    
            candidates.append((i,speed,rotational_speed))
        else:
            rejected.append((frames[i].index, frames[i].skewness, scale_factor_change, speed, rotational_speed))
            
        
    if len(candidates) > 0:
        # Choose the frames with lowest average skewness
        i = np.argmin(skews)
        return candidates[i]
    else:
        print('Error! No suitable frames found')
        print('Min skewness', np.min(skews))
        print('Rejected frames are:')
        for idx, skew, scale, speed, rot in rejected:
            print(idx, skew, scale, speed, rot)
      
        return None, [], None

if __name__ == '__main__':
    # Example usage
    video_path = 'c:/keg/tracking/GX010375.mp4'
    
    disc_id, frames, time_per_frame = analyze_frames(video_path, output_path='c:/keg/tracking/tracking')
    
    if disc_id is not None:
        print(f'Disc ID {disc_id}')
    if frames:
        i, traveling_speed, rotational_speed = calculate_speeds(frames, time_per_frame, nmin=0)
        if i is not None:
            print(f'Calculations based on detected frames {i-1}/{i}')
            print(f'Traveling speed:  {traveling_speed:.2f} m/s       {traveling_speed*2.237:.2f} mph')
            print(f'Rotational speed: {rotational_speed:.2f} rad/s    {rotational_speed*60/(2*np.pi):.2f} RPM')
    else:
        print('Could not detect disc...')
        

