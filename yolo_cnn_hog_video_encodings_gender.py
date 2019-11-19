from __future__ import division
import os
import configparser
import pickle
import psutil
import time
import datetime
import numpy as np
import pandas as pd 
import sklearn
import cv2
import imutils
import dlib
import face_recognition
import torch 
import torch.nn as nn
from torch.autograd import Variable
from util import load_classes, write_results
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import cluster_1_door

config = configparser.ConfigParser()
config.read('cfg.ini')

# All parameters that could be changed on daily usage
cwd = config['context']['cwd']
i_frame = 1 # Number of frame in video to begin processing
debug = False # Whether to show images or not
date = str(datetime.datetime.now().date())
small_frame_w = 768 # Width of all small frames used
padding = 30 # Padding for contours and drawn faces
face_padding = 15 # Padding used to improve gender detection
hog_threshold = -0.4 # Score threshold for HOG face detector
temp_threshold = 96 # Threshold of CPU temperature to enter time.sleep
t0 = time.time()

############################################################
##### FOR YOLO #############################################
############################################################
confidence = float(config['yolo']['confidence']) #float(args.confidence)
nms_thesh = 0.4 #float(args.nms_thresh)
CUDA = torch.cuda.is_available()
num_classes = 80
# bbox_attrs = 5 + num_classes

print("Loading network.....")
model = Darknet(cwd + "cfg/yolov3.cfg")
model.load_weights(cwd + "yolov3.weights")
classes = load_classes(cwd + 'data/coco.names')
print("Network successfully loaded")
model.net_info["height"] = config['yolo']['height']
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
if CUDA:
    model.cuda()
    
############################################################
############################################################
############################################################

# HOG face detector
hog = dlib.get_frontal_face_detector()

# Load gender model. 
# https://www.learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/
genderProto = cwd + 'gender_deploy.prototxt'
genderModel = cwd + 'gender_net.caffemodel'
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

rests = 0

# Load pre-trained random forest to distinguish faces from wheels (using the
# mean of RGB channels)
with open(cwd + 'forest.pickle', 'rb') as f:
    forest = pickle.load(f)

print('Loaded', type(forest))

# Create directory for frames
if not (os.path.isdir(cwd + 'frames_cnn_hog/' + date) or debug):
    os.mkdir(cwd + 'frames_cnn_hog/' + date)

# Create faces DataFrame and encodings array
df_faces = []
encodings = []

# Get video capture of "video_[date].avi" (verbosely)
cap = cv2.VideoCapture(cwd + 'videos/' + date + '.avi')

print('W:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('H:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('FPS:', cap.get(cv2.CAP_PROP_FPS))
print(date)

# Start at i_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame - 1)
n_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Starting at frame', n_frame, 'of', total_frames)

while True:
    t1 = time.time()    

    # Set person and frontal face boolean as False and face_in_frame as 1
    person = False
    frontal_face = False
    face_in_frame = 1

    ret, frame = cap.read()
    # Break if not correctly read
    if not ret:
        break 

    small_frame = imutils.resize(frame, width=small_frame_w)
    r = frame.shape[1] / small_frame.shape[1]
    
    ############################################################
    ##### FOR YOLO #############################################
    ############################################################ 
    orig_im = small_frame.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    im_dim = torch.FloatTensor(dim).repeat(1,2)
    if CUDA:
       im_dim = im_dim.cuda()
       img = img.cuda()
    with torch.no_grad():   
        output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, 
                           nms_conf = nms_thesh)
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

    ppl_left, ppl_top, ppl_right, ppl_bottom = (small_frame.shape[1], 
                                                small_frame.shape[0], 0, 0)

    for thing in output:
        cls = int(thing[-1])
        label = "{0}".format(classes[cls])
        bbox = thing[1:5].int()
        if (label == 'person') and (bbox.prod().item() != 0):
            pleft, ptop, pright, pbottom = bbox
            pleft, ptop, pright, pbottom = (pleft.item(), ptop.item(), 
                                            pright.item(), pbottom.item())
    
            ppl_left = min(ppl_left, pleft)
            ppl_top = min(ppl_top, ptop)
            ppl_right = max(ppl_right, pright)
            ppl_bottom = max(ppl_bottom, pbottom)
            person = True
            
    ############################################################
    ############################################################
    ############################################################
    
    if person:
        # print('PROCESSING')
        ppl_top = max(ppl_top - padding, 0)
        ppl_right = min(ppl_right + padding, int(small_frame.shape[1]))
        # ppl_bottom = min(ppl_bottom + padding, int(small_frame.shape[0]))
        ppl_left = max(ppl_left - padding, 0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        ppl_frame = rgb_small_frame[ppl_top: ppl_bottom, 
                                    ppl_left: ppl_right]

        face_locations = face_recognition.face_locations(ppl_frame, 
                                                         model = "cnn")
        # Convert face locations from contour coordinates to small_frame
        # coordinates and append to the list
        face_locations = [tuple([face[0] + ppl_top, face[1] + ppl_left, 
                                 face[2] + ppl_top, face[3] + ppl_left]) 
                          for face in face_locations]
        big_face_locations = [tuple([int(r*x) for x in face]) for 
                              face in face_locations]

        face_encodings = face_recognition.face_encodings(
            rgb_frame, big_face_locations)

        for face_location, big_face_location, face_encoding in zip(
                face_locations, big_face_locations, face_encodings):
            # print("FACE")
            top, right, bottom, left = face_location
            big_top, big_right, big_bottom, big_left = big_face_location

            top = max(top - padding, 0)
            right = min(right + padding, int(rgb_small_frame.shape[1]))
            bottom = min(bottom + padding, int(rgb_small_frame.shape[0]))
            left = max(left - padding, 0)

            hog_fl, scores, idx = hog.run(rgb_small_frame[top:bottom, 
                                                          left:right], 
                                          1, hog_threshold)

            if len(hog_fl):
                # print('FRONTAL')
                rgb = rgb_frame[big_top:big_bottom, 
                                big_left:big_right, 
                                :].mean(axis=0).mean(axis=0)

                # Prepare RGB features for Random Forest
                f_rgb = list(rgb)    
                f_rgb = f_rgb + [f_rgb[2] - f_rgb[0], f_rgb[1] - f_rgb[0], 
                             f_rgb[2] - f_rgb[1]]
                f_rgb = np.array(f_rgb, dtype=np.float32).reshape(1, -1)
                forest_pred = int(forest.predict(f_rgb)[0])

                if forest_pred:
                    # print('NOT A WHEEL')
                    h_face = big_bottom - big_top

                    face_top = max(0,big_top-face_padding)
                    face_bottom = min(big_bottom+face_padding, 
                                      frame.shape[0]-1)
                    face_left = max(0,big_left-face_padding)
                    face_right = min(big_right+face_padding, 
                                     frame.shape[1]-1)

                    face = frame[face_top: face_bottom,
                                 face_left:face_right]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                                 MODEL_MEAN_VALUES, 
                                                 swapRB=False)

                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    gender_conf = genderPreds[0].max()
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(small_frame, str(face_in_frame), 
                                (left + 6, bottom - 6), 
                                font, 0.8, (255, 255, 255), 1)
                    cv2.putText(small_frame, str(gender), 
                                (left, bottom + 24), font, 0.8, (255, 0, 0))
                    cv2.rectangle(small_frame, (left, top), (right, bottom), 
                                  (0,0,255), 2)

                    df_faces.append({'date': date, 'frame': n_frame, 
                                     'face_in_frame': face_in_frame,
                                     'h_face': h_face, 'RGB': rgb, 
                                     'score': scores[0], 
                                     'face_type': idx[0], 'gender': gender,
                                     'gender_conf': gender_conf})
                    encodings.append(face_encoding)
                    
                    frontal_face = True
                    face_in_frame += 1

    if frontal_face and not debug:
        cv2.imwrite(
            (cwd + 'frames_cnn_hog/' + date + '/'
             + str(n_frame) + '.jpg'), 
            small_frame)

    if debug:
        if person:
            cv2.rectangle(small_frame, (ppl_left, ppl_top),
                          (ppl_right, ppl_bottom), (255,255,255), 1, 8)

        cv2.imshow('procesado', small_frame)    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if (psutil.sensors_temperatures()['coretemp'][0].current 
            > temp_threshold):    
        # print('resting...')
        while (psutil.sensors_temperatures()['coretemp'][0].current > 68):
            time.sleep(1)
        rests += 1

    # print(round(time.time()-t1, 4), round(n_frame/total_frames * 100, 2))
    n_frame += 1
    
df_faces = pd.DataFrame(df_faces)
encodings = np.array(encodings)

if not debug:
    np.save((cwd + 'encodings_cnn_hog/' + date + '_' + str(i_frame)
             + '.npy'), encodings)
    df_faces = df_faces[['date', 'frame', 'face_in_frame', 'h_face', 'RGB',
                         'score', 'face_type', 'gender', 'gender_conf']]
    df_faces.to_csv(
        cwd + 'CSVs_cnn_hog/' + date + '_' + str(i_frame) + '.csv')
    print('CSV and encodings saved')

cap.release()

if debug:
    cv2.destroyAllWindows()

print('Processed', n_frame, 'of', total_frames, 'frames')
print('Faces data shapes:', df_faces.shape, encodings.shape)
print('Total time:', round((time.time() - t0) / 3600, 2), 'hours')
print('Rested ' + str(rests) + ' times')
print('Done processing', date)

cluster_1_door.main(date)
