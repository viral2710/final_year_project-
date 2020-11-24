import cv2
s='driving_dataset/data/0.jpg'
read_image = cv2.imread(s)
print(read_image.shape)
cv2.imshow('xyz',read_image)
cv2.waitKey(0)
cv2.destroyAllWindows()