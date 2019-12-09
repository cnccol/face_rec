import os
import configparser
import time
import datetime
import cv2
import imutils

config = configparser.ConfigParser()
config.read(os.path.expanduser('~/Desktop/cfg.ini'))

cwd = config['context']['cwd']
cam_index = int(config['record']['cam_index'])

tf_hour = 20
tf_minute = 0

now = datetime.datetime.now()
print('Recording', now)

date = str(now.date())

cap = cv2.VideoCapture(cam_index)
time.sleep(2.0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')    
out = cv2.VideoWriter(cwd + 'videos/' + date + '.avi',fourcc, 12.5, (1920,1080))

cuadro = 1

try:
    while True:
        tic = time.time()

        ret, frame = cap.read()
       
        out.write(frame)

        timestamp = datetime.datetime.fromtimestamp(tic)
        if (timestamp.hour == tf_hour) and (timestamp.minute == tf_minute):
            break
        
        toc = time.time()

        # Impresión de control para el log, cuando está grabando bien está
        # alrededor de 0.03-0.04 (excepto la primera iteración que es mayor
        if cuadro % 1500 == 1:
            print(toc - tic)
        
        cuadro += 1

        # Lograr 12.5 FPS (1 / 12.5 = 0.08 SPF) en realidad
        try:
            time.sleep(0.08 - (toc - tic))
        except:
            pass	

# Éste try-except es para que, en caso de error, se logren los release. En caso
# de que el error no sea bien "agarrado" por acá, arreglar el video con:
# ffmpeg -i <input.avi> -c copy <output.avi>
except Exception as e:
    print(e)
    pass

cap.release()
out.release()
print(date, 'done')
print()
