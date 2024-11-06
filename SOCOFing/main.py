import os
import cv2

sample = cv2.imread("Real/34__M_Left_little_finger.BMP")

# sample = cv2.resize(sample, None, fx=2.5, fy= 2.5)

# cv2.imshow("Sample", sample)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

image = None
fileName = None
kp1, kp2, mp = None, None, None
bestscore = 0

for file in os.listdir("Real"):
    fingerprint_image = cv2.imread(f"Real/{file}")
    sift = cv2.SIFT_create() #initialize the sift method

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
    

    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}).knnMatch(descriptors_1, descriptors_2, k = 2)

    match_point = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_point.append(p)

    
    keypoints = min(len(keypoints_1), len(keypoints_2))
    if keypoints == 0:
        continue 
    
    score = (len(match_point) / keypoints) * 100
    if score > bestscore:
        bestscore = score
        fileName = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_point


print("Best Match: " + fileName)
print("Score: " + str(bestscore))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
cv2.resize(result,None, fx=4, fy= 4)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
