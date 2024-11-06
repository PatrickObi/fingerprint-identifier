import os
import cv2 as cv

sample = cv.imread("Real/1__M_Right_little_finger.BMP")

bestscore = 0
filename = None

for file in os.listdir("Real/"):
    fingerprint_image = cv.imread("Real/" + file)

    print("Processing: " + file)

    sift = cv.SIFT_create()

    keypoint1, descriptors1 = sift.detectAndCompute(sample, None)
    keypoint2, descriptors2 = sift.detectAndCompute(fingerprint_image, None)

    keypoints = min(len(keypoint1), len(keypoint2))
    if keypoints == 0:
        continue

    matches = cv.FlannBasedMatcher({"algorithm": 1, "trees": 10}).knnMatch(descriptors1, descriptors2, k = 2)

    best_matches = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            best_matches.append(p)

    score = len(best_matches) /keypoints * 100
    if score > bestscore:
        filename = file
    
if __name__ == "__main__":
    if filename != None:
        print("Sorry you have already voted")

