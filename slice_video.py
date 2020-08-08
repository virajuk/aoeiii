import cv2

cap = cv2.VideoCapture("/home/viraj-uk/Videos/age3y_4.mp4")

count = 1
while True:
    ret, image_np = cap.read()

    cv2.imwrite("/home/viraj-uk/Pictures/age3y_4/"+str(count)+".jpeg", image_np)
    # cv2.imshow("video", image_np)

    count+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break