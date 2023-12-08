import time
import cv2
import imutils

# initialize the motion saliency object and start the video stream
saliency = None
cap = cv2.VideoCapture(0)
tm = cv2.TickMeter()
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    _, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    # if our saliency object is None, we need to instantiate it
    if saliency is None:
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014.create()
        saliency.setImagesize(frame.shape[1], frame.shape[0])
        saliency.init()

    # convert the input frame to grayscale and compute the saliency
    # map based on the motion model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tm.start()
    (success, saliencyMap) = saliency.computeSaliency(gray)
    tm.stop()

    saliencyMap = (saliencyMap * 255).astype("uint8")

    cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # display the image to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("Map", saliencyMap)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.stop()
