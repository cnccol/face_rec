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
import cluster_1_door

config = configparser.ConfigParser()
config.read(os.path.expanduser('~/Desktop/cfg.ini'))

cwd = config['context']['cwd']

# All parameters that could be changed on daily usage
def key_params():
    i_frame = 1 # Number of frame in video to begin processing
    see = False # Whether to show images or not
    date = str(datetime.datetime.now().date())
    small_frame_w = 768 # Width of all small frames used
    h_line_1 = int(config['cnn_hog']['h_line_1']) # Top position of stripe to check movement in
    h_line_2 = int(config['cnn_hog']['h_line_2']) # Bottom of said stripe
    n_last_frames = 4 # How many frames ago to compare (simple_motion_detector)
    delta_threshold = 10 # Difference threshold for simple_motion_detector
    dilate_iterations = 16 # fgmask dilation (simple_motion_detector)
    min_stripe_area = 450 # Minimum movement area in strip for mov=True
    padding = 30 # Padding for contours and drawn faces
    face_padding = 15 # Padding used to improve gender detection
    hog_threshold = -0.4 # Score threshold for HOG face detector
    temp_threshold = 96 # Threshold of CPU temperature to enter time.sleep

    return (i_frame, see, date, small_frame_w, h_line_1, h_line_2, 
            n_last_frames, delta_threshold, dilate_iterations, min_stripe_area,
            padding, face_padding, hog_threshold, temp_threshold)

# https://www.learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/
def load_gender_model():
    genderProto = cwd + 'gender_deploy.prototxt'
    genderModel = cwd + 'gender_net.caffemodel'

    genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    genderList = ['Male', 'Female']

    return genderNet, MODEL_MEAN_VALUES, genderList

# Load pre-trained random forest to distinguish faces from wheels (using the
# mean of RGB channels)
def load_forest():
    with open(cwd + 'forest.pickle', 'rb') as f:
        forest = pickle.load(f)
    print('Loaded', type(forest))

    return forest

# Initialize video capture (printing video information and setting i_frame)
def get_video_capture(date, i_frame):
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

    return n_frame, total_frames, cap

def simple_motion_detector(mov_frame, n_frame, i_frame, n_last_frames, 
                           last_frames, delta_threshold, dilate_iterations):
    
    mov_frame = cv2.cvtColor(mov_frame, cv2.COLOR_BGR2GRAY)
    mov_frame = cv2.GaussianBlur(mov_frame, (15, 15), 0)

    # In first n_last_frames, act as if no movement and construct the list
    # last_frames
    if n_frame in [i_frame + k for k in range(n_last_frames)]:
        last_frames.append(mov_frame)
        fgmask = np.zeros_like(mov_frame)

        return fgmask, last_frames
    
    # Compare current frame to the one n_last_frames ago
    fgmask = cv2.absdiff(mov_frame, last_frames[0])
    fgmask = cv2.threshold(fgmask, delta_threshold, 255, cv2.THRESH_BINARY)[1]
    fgmask = cv2.dilate(fgmask, None, iterations=dilate_iterations)

    # Update last_frames
    last_frames = last_frames[1:]
    last_frames.append(mov_frame)

    return fgmask, last_frames

# Use fgmask to check if there's movement in the stripe 
def check_stripe_movement(fgmask, min_stripe_area, h_line_1, h_line_2):
    mov = False
    
    # Get contours of fgmask inside fgmask
    cnts = cv2.findContours(fgmask.copy()[h_line_1: h_line_2, :], 
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # mov becomes true if any contour is big enough
    for c in cnts:
        if cv2.contourArea(c) < min_stripe_area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(fgmask, (x, y + h_line_1), 
                              (x + w, y + h + h_line_1), 100, 2)
        mov = True
    
    return mov

def get_forest_pred(forest, rgb):
    # Prepare RGB features for Random Forest
    rgb = list(rgb)    
    rgb = rgb + [rgb[2] - rgb[0], rgb[1] - rgb[0], rgb[2] - rgb[1]]
    rgb = np.array(rgb, dtype=np.float32).reshape(1, -1)
    pred = int(forest.predict(rgb)[0])

    return pred

def hog_face_loc(rgb_small_frame, top, right, bottom, left, padding,
                 hog, hog_threshold):
    top = top - padding
    right = right + padding
    bottom = bottom + padding
    left = left - padding

    htop = max(top, 0)
    hright = min(right, int(rgb_small_frame.shape[1]))
    hbottom = min(bottom, int(rgb_small_frame.shape[0]))
    hleft = max(left, 0)

    dets, scores, idx = hog.run(rgb_small_frame[htop:hbottom, 
                                                hleft:hright], 
                                1, hog_threshold)
    
    return dets, scores, idx
    
def predict_gender(frame, big_top, big_right, big_bottom, big_left, 
                   face_padding, MODEL_MEAN_VALUES, genderNet, genderList):
    
    face = frame[max(0,big_top-face_padding):min(big_bottom+face_padding, 
                                                 frame.shape[0]-1),
                 max(0,big_left-face_padding):min(big_right+face_padding, 
                                                  frame.shape[1]-1)]

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, 
                                 swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    gender_conf = genderPreds[0].max()
    
    return gender, gender_conf

def draw_face_small_frame(small_frame, face_in_frame, gender, top, right, 
                          bottom, left, padding):
    # Draw rectangles and face_in_frame text with padding so face is visible
    top = top - padding
    right = right + padding
    bottom = bottom + padding
    left = left - padding
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(small_frame, str(face_in_frame), (left + 6, bottom - 6), 
                font, 0.8, (255, 255, 255), 1)
    cv2.putText(small_frame, str(gender), (left, bottom + 24), font, 0.8, 
                (255, 0, 0))
    cv2.rectangle(small_frame, (left, top), (right, bottom), (0,0,255), 2)

def draw_and_show(small_frame, fgmask, h_line_1, h_line_2):
    w = small_frame.shape[1]
    cv2.line(small_frame, (0, h_line_1), (w, h_line_1), 
             (255, 255, 255))
    cv2.line(small_frame, (0, h_line_2), (w, h_line_2), 
             (255, 255, 255))
    cv2.imshow('procesado', small_frame)
    cv2.imshow('movimiento', fgmask)
    
    key = cv2.waitKey(1) & 0xFF

    return key      

def main():
    t0 = time.time()
    
    # Initialize empty list of frames for simple_motion_detector
    last_frames = []

    # HOG face detector
    hog = dlib.get_frontal_face_detector()

    # Load gender model
    genderNet, MODEL_MEAN_VALUES, genderList = load_gender_model()

    # Key parameters
    (i_frame, see, date, small_frame_w, h_line_1, h_line_2, n_last_frames, 
     delta_threshold, dilate_iterations, min_stripe_area, padding, 
     face_padding, hog_threshold, temp_threshold) = key_params()

    # Counters: one for the case where a frame has more than one face, the
    # other for counting how many times the computer has slept to avoid
    # overheating
    face_in_frame, rests = 1, 0

    # Load random forest classifier (get rid of wheel detections)
    forest = load_forest()

    # Create directory for frames
    if not (os.path.isdir(cwd + 'frames_cnn_hog/' + date) or see):
        os.mkdir(cwd + 'frames_cnn_hog/' + date)

    # Create faces DataFrame and encodings array
    df_faces = pd.DataFrame(columns = ['date', 'frame', 'face_in_frame', 
                                       'h_face', 'RGB', 'score', 'face_type', 
                                       'gender', 'gender_conf'])
    encodings = np.zeros((0, 128))
    
    # Get video capture of "video_[date].avi" (verbosely)
    n_frame, total_frames, cap = get_video_capture(date, i_frame)

    while True:
        
        # Set frontal face boolean as False and face_in_frame as 1
        frontal_face = False
        face_in_frame = 1
        
        n_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)

        # Read frame
        ret, frame = cap.read()

        # Break if not correctly read
        if not ret:
            break 
        
        small_frame = imutils.resize(frame, width=small_frame_w)
        r = frame.shape[1] / small_frame.shape[1]

        fgmask, last_frames = simple_motion_detector(
            small_frame.copy(), n_frame, i_frame, n_last_frames, 
            last_frames, delta_threshold, dilate_iterations)

        mov = check_stripe_movement(fgmask, min_stripe_area, h_line_1, 
                                    h_line_2)

        if mov:
            # print('PROCESSING')

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame, 
                                                             model = "cnn")

            big_face_locations = [tuple([int(r*x) for x in face]) for 
                                  face in face_locations]

            face_encodings = face_recognition.face_encodings(
                rgb_frame, big_face_locations)

            for face_location, big_face_location, face_encoding in zip(
                    face_locations, big_face_locations, face_encodings):
                # print("FACE")
                top, right, bottom, left = face_location

                big_top, big_right, big_bottom, big_left = big_face_location

                rgb = rgb_frame[big_top:big_bottom, 
                                big_left:big_right, 
                                :].mean(axis=0).mean(axis=0)

                forest_pred = get_forest_pred(forest, rgb)

                h_face = big_bottom - big_top

                hog_fl, scores, idx = hog_face_loc(
                    rgb_small_frame, top, right, bottom, left, padding, 
                    hog, hog_threshold)

                if len(hog_fl) and forest_pred:
                    # print('FRONTAL')

                    gender, gender_conf = predict_gender(
                        frame, big_top, big_right, big_bottom, big_left,
                        face_padding, MODEL_MEAN_VALUES, genderNet, genderList)

                    draw_face_small_frame(small_frame, face_in_frame, 
                                            gender, top, right, bottom, 
                                            left, padding)

                    df_faces = df_faces.append({'date': date, 
                                                'frame': n_frame, 
                                                'face_in_frame': face_in_frame,
                                                'h_face': h_face, 
                                                'RGB': rgb, 
                                                'score': scores[0], 
                                                'face_type': idx[0],
                                                'gender': gender,
                                                'gender_conf': gender_conf},
                                               ignore_index=True)

                    encodings = np.concatenate(
                        (encodings, face_encoding.reshape((1, 128))))
                    
                    frontal_face = True
                    face_in_frame += 1

        if frontal_face and not see:
            cv2.imwrite(
                cwd + 'frames_cnn_hog/' + date + '/' + str(n_frame) + '.jpg', 
                small_frame)    

        if see:
            key = draw_and_show(small_frame, fgmask, h_line_1, h_line_2)
            if key == ord('q'):
                break  

        if ((n_frame % 10000) == 0) and not see:
            df_faces.to_csv(
                cwd + 'CSVs_cnn_hog/' + date + '_' + str(i_frame) + '.csv')
            np.save((cwd + 'encodings_cnn_hog/' + date + '_' + str(i_frame)
                     + '.npy'), encodings)
            # print('-----------CSV and encodings saved------------')

        if (psutil.sensors_temperatures()['coretemp'][0].current
                > temp_threshold):
            # print('resting...')
            while (psutil.sensors_temperatures()['coretemp'][0].current > 68):
                time.sleep(1)
            rests += 1

    if not see:
        np.save((cwd + 'encodings_cnn_hog/' + date + '_' + str(i_frame)
                 + '.npy'), encodings)
        df_faces.to_csv(
            cwd + 'CSVs_cnn_hog/' + date + '_' + str(i_frame) + '.csv')
        print('CSV and encodings saved')

    cap.release()

    if see:
        cv2.destroyAllWindows()

    print('Processed', n_frame, 'of', total_frames, 'frames')
    print('Faces data shapes:', df_faces.shape, encodings.shape)
    print('Total time:', round((time.time() - t0) / 3600, 2), 'hours')
    print('Rested ' + str(rests) + ' times')
    print('Done processing', date)

    cluster_1_door.main(date)


if __name__ == '__main__':
    main()
