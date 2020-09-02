import glob
import os
import cv2

images = glob.glob("/home/viraj-uk/Desktop/th_3_british/eval/*.png")
save_path = "/home/viraj-uk/Pictures/th_3_british/eval"

count = 1
for image in images:

    # image_name = os.path.basename(image)
    # print(image_name)
    cv_image = cv2.imread(image, cv2.COLOR_BGR2RGB)

    save_name = str(count)+".jpeg"
    cv2.imwrite(os.path.join(save_path, save_name), cv_image)

    count += 1


    # image_name = os.path.basename(image)
    # print(image_name)

    # cv2.imshow('Preview', cv_image)
    # cv2.waitKey()

# cv2.destroyWindow()