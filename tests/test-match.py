import os
import cv2
from fscan2.match import match_all_circles_normalized, match_big_circles
from fscan2.detect_circles import detect_circles
from fscan2.visualization import plot_circles

def plot_big_circles(img1, img2, circles1, circles2):
    circles1_big, circles2_big = match_big_circles(circles1, circles2)
    output1 = plot_circles(img1, circles1_big, counting=True)
    output2 = plot_circles(img2, circles2_big, counting=True)
    return output1, output2


def plot_all_circles(img1, img2, circles1, circles2):
    matched_circles1, matched_circles2, dist = match_all_circles_normalized(circles1, circles2)
    output1 = plot_circles(img1, matched_circles1, counting=True)
    output2 = plot_circles(img2, matched_circles2, counting=True)
    return output1, output2
    

def load_images(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print('图片读取失败')
    return img1, img2

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, 'tests')
    os.makedirs(tests_dir, exist_ok=True)  # Create tests directory if it doesn't exist
    os.chdir(tests_dir)  # Change working directory to tests
    img1_path = os.path.join(tests_dir, '0006.png')
    img2_path = os.path.join(tests_dir, '0007.png')

    print(f'读取图片: {img1_path}\n读取图片: {img2_path}')
    img1, img2 = load_images(img1_path, img2_path)

    circles1 = detect_circles(img1)
    circles2 = detect_circles(img2)

    # image1, image2 = plot_big_circles(img1, img2, circles1, circles2)
    image1, image2 = plot_all_circles(img1, img2, circles1, circles2)
    
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(tests_dir, 'circles1.png'), image1)
    cv2.imwrite(os.path.join(tests_dir, 'circles2.png'), image2)
    # plot_all_circles(img1, img2, circles1, circles2)