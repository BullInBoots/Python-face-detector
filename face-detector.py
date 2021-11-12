import cv2

trained_face_data = cv2.CascadeClassifier(C:\Users\Phern\Desktop\Finn's Folder\dot-xml-files)

img = cv2.imread("RDJ.jpg")

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)
# cv2.imshow('Face', grayscaled_img)
cv2.waitKey()
print("Code completed")