# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import easyocr

matplotlib.rc("image", cmap="gray")

# %%
# %matplotlib qt

# %%
image = cv2.imread(r"Doc.jpg")
plt.imshow(image)

# %%
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
# result_image = cv2.bitwise_and(gray_image, mask)
plt.imshow(mask)

# %%
reader = easyocr.Reader(lang_list=['ru'])

# %%
BORDER_COLOR = (255, 0, 255)
show_image = image.copy()
contours = reader.detect(img=mask, width_ths=0.5)[0][0]
for contour in contours:
    x0, x1, y0, y1 = contour
    cv2.rectangle(show_image, (x0, y0), (x1, y1), BORDER_COLOR, 1)

plt.imshow(show_image)
