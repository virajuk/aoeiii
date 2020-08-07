import cv2

cap = cv2.VideoCapture("/home/viraj-uk/Documents/aoe3_bck/video/aoe_iii_snow_hills.mp4")

while True:
    ret, image_np = cap.read()

    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break