import cv2

cap = cv2.VideoCapture("/home/viraj-uk/Videos/bighorn_sheep_blue.mp4")

count = 1
while True:
    ret, image_np = cap.read()

    cv2.imwrite("/home/viraj-uk/Pictures/bighorn_sheep_blue/"+str(count)+".jpeg", image_np)
    # cv2.imshow("video", image_np)

    count+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break