import cv2
import numpy as np

def plot_circles(img, circles, counting=False):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in circles:
        cv2.circle(color_img, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 1)
        cv2.circle(color_img, (int(c[0]), int(c[1])), 2, (0, 0, 255), 1)
    if counting:
        for i, c in enumerate(circles):
            cv2.putText(color_img, str(i+1), (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return color_img

def plot_matched_circles(src_img, src_circles, matched_circles):
    color_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

    H, _ = cv2.findHomography(np.array([[c[0], c[1]] for c in matched_circles]),
                              np.array([[c[0], c[1]] for c in src_circles]))
    scale = np.linalg.norm(H[0, :2])
    for c in matched_circles:
        pt = np.array([[[c[0], c[1]]]], dtype=np.float32)
        x2, y2 = cv2.perspectiveTransform(pt, H)[0][0].astype(int)
        r = int(c[2] * scale)  
        cv2.circle(color_img, (x2, y2), r, (0, 0, 255), 1)
    return color_img
