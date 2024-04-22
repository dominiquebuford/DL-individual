import numpy as np
import cv2

def display_keypoints(keypoints, ganImage = False, imageID = 0):

    if ganImage:
        image = cv2.imread(f"captures/new_image_{imageID}.png")
    else:
        image = cv2.imread(f"captures/image_{imageID}.png")  
    
    if isinstance(keypoints, str):
        cv2.imwrite(f'static/keypoint_image_{imageID}.jpg', image)
        return

    print(keypoints)
    # Iterate over each keypoint and draw a circle on the image
    for kp in keypoints:
        x, y, c = kp
        if c>0.07:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw a green circle

    cv2.imwrite(f'static/keypoint_image_{imageID}.jpg', image)