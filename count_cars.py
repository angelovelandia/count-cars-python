import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('autos.mp4')

# charge model
cars = cv2.CascadeClassifier('cars.xml')

# extract bg
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

car_counter = 0

while True:

    # extract frame
    ret, frame = cap.read()
    if ret == False: break

    # resize video frames
    frame = imutils.resize(frame, width=640)

    # points area to analize
    area_point = np.array([[330, 216], [frame.shape[1]-80, 216], [frame.shape[1]-80, 271], [330, 271]])

    # img aux
    imgAux = np.zeros(shape=(frame.shape[:2]), dtype = np.uint8)
    imgAux = cv2.drawContours(imgAux, [area_point], -1, (255), -1)
    # area video
    imgArea = cv2.bitwise_and(frame, frame, mask=imgAux)

    # subtrack bg
    fgmask = fgbg.apply(imgArea)
    # better img
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=5)

    # find contours
    conTours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # contours for
    for cont in conTours:
        if cv2.contourArea(cont) > 1500:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)

            roiGray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            #cv2.imshow('roiGray', roiGray)
            detected = cars.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=1)

            if len(detected) > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                if 430 < (x + w) < 470:
                    car_counter = car_counter + 1
                    cv2.line(frame, (450, 216), (450, 271), (0, 255, 0), 3)

    # view area
    cv2.drawContours(frame, [area_point], -1, (255, 0, 255), 2)
    # draw line
    cv2.line(frame, (450, 216), (450, 271), (0, 255, 255), 1)
    # draw counter
    cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0, 255, 0), 2)
    cv2.putText(frame, str(car_counter), (frame.shape[1]-55, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # show frame
    cv2.imshow('Frame', frame)
    # show imgAux
    #cv2.imshow('fgmask', fgmask)

    # wait a key ESC
    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()