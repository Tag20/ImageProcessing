#AIM: Write a program to apply various (Hadamard and Walsh)  transforms on an image and compare the results. Also, perform inverse operations after blacking out quadrants

import cv2
from google.colab.patches import cv2_imshow
from scipy.linalg import hadamard
import numpy as np

image = cv2.imread('/content/hi.png')
# cv2_imshow(image)
new_width = 256
new_height = 256

# Resize the image
img = cv2.resize(image, (new_width, new_height))
# cv2_imshow(image)
# cv2_imshow(img)

grayscale_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2_imshow(grayscale_img)

image_size = max(grayscale_img.shape)

image_size

"""## Hadamard



"""

#hadamard
hadamard_matrix=hadamard(image_size)
hadamard_matrix

hadamard_transformed_1 = np.dot(hadamard_matrix,grayscale_img)
final = np.dot(hadamard_transformed_1,hadamard_matrix.T)

cv2_imshow(final)

hadamard_matrix_inv_1 = np.dot(hadamard_matrix,final)
inverse_final = np.dot(hadamard_matrix_inv_1,hadamard_matrix.T)
inverse_final=inverse_final//(256*256)

cv2_imshow(inverse_final)

quad_4=final.copy()
quad_4[127:,127:]=0
cv2_imshow(quad_4)

quad_4_inv = np.dot(hadamard_matrix,quad_4)
quad_4_inv_final = np.dot(quad_4_inv,hadamard_matrix.transpose())
quad_4_inv_final=quad_4_inv_final/(quad_4.shape[0]**2)
cv2_imshow(quad_4_inv_final)

quad_3_4=final.copy()
quad_3_4[127:,:]=0
cv2_imshow(quad_3_4)

quad_3_4_inv = np.dot(hadamard_matrix,quad_3_4)
quad_3_4_inv_final = np.dot(quad_3_4_inv,hadamard_matrix.transpose())
quad_3_4_inv_final=quad_3_4_inv_final/(quad_3_4.shape[0]**2)
cv2_imshow(quad_3_4_inv_final)

quad_2_3_4=final.copy()
quad_2_3_4[127:]=0
quad_2_3_4[:,127:]=0
cv2_imshow(quad_2_3_4)

quad_2_3_4_inv = np.dot(hadamard_matrix,quad_2_3_4)
quad_2_3_4_inv_final = np.dot(quad_2_3_4_inv,hadamard_matrix.transpose())
quad_2_3_4_inv_final=quad_2_3_4_inv_final/(quad_2_3_4.shape[0]**2)
cv2_imshow(quad_2_3_4_inv_final)

"""# Walsh"""

H2 = np.array([[1, 1], [1, -1]])
H4 = np.kron(H2, H2)

H4

def sums(row):
  sum = 0
  for i in range(len(row)-1):
    if row[i] != row[i+1]:
      sum = sum + 1
  return sum

H = np.array([[1, 1], [1, -1]])
for i in range(7):
  H = np.kron(H,H2)
print("Matrix Size:",H.shape)
print(H)

walsh_matrix = np.array(sorted(H, key=sums))

final = np.matmul(np.matmul(walsh_matrix,grayscale_img),walsh_matrix)
cv2_imshow(final)
final_inverse = np.matmul(np.matmul(walsh_matrix,final),walsh_matrix)/(256 * 256)
cv2_imshow(final_inverse)
print(" Checking the array:\n",final_inverse - grayscale_img)
