import numpy
import qimage2ndarray
from PyQt5.QtGui import QPainter, QImage
from PyQt5.Qt import Qt
import cv2
import warnings

from tqdm import tqdm

# constant from game
screenWidth = 800
screenHeight = 600

fieldHeight = 50
fieldColor = Qt.green
barHeight = 160
barWidth = 20
barPosX = (screenWidth // 2) - (barWidth // 2)
barPosY = screenHeight - fieldHeight - barHeight

player1limitX = (0, barPosX - barWidth - 60)     
player2limitX = (barPosX + barWidth , screenWidth - barWidth)

player1AngleLimit = (0, 84)
player2AngleLimit = (96, 180)

playerHeight = barHeight // 3 * 2
playerWidth = barWidth

playerY = screenHeight - fieldHeight - playerHeight

playerColor = Qt.black
barColor = Qt.blue

stoneHeight = 10
stoneWidth = 10
stoneColor = Qt.red
stoneY = playerY - stoneHeight

# sets the value of the imageCrop variable to 300 
imageCrop = 300

imageResize = (80,30)

# loads data from the "modelData.npy" file and stores it in the all_data variable.
all_data = numpy.load('modelData.npy')

img_data = []
output = []

# creates a QImage object
image = QImage(screenWidth, screenHeight, QImage.Format_RGB32)

# painter that will be used to draw on the image.
painter = QPainter(image)

for data in tqdm(all_data):
    target_dist = data[0]
    bar_dist = data[1]
    
    player1X = int(barPosX - bar_dist)
    player2X = int(player1X + target_dist)
    stoneX = player2X + playerWidth - stoneWidth
    
    #draw
    image.fill(Qt.white)

    painter.fillRect(0, screenHeight - fieldHeight,
                           screenWidth, fieldHeight, fieldColor)
    painter.fillRect(barPosX - 60, barPosY, barWidth,
                  barHeight, barColor)
    painter.fillRect(barPosX, barPosY, barWidth,
                  barHeight, barColor)
    painter.fillRect(player1X, playerY, playerWidth,
                      playerHeight, playerColor)
    painter.fillRect(player2X, playerY, playerWidth,
                      playerHeight, playerColor)
    painter.fillRect(stoneX, stoneY, stoneWidth,
                      stoneWidth, stoneColor)

#   converts the QImage object image to a NumPy array
    arr_image = qimage2ndarray.byte_view(image)


    cropped_image = arr_image[imageCrop:]

#   resizes the cropped image to the dimensions specified by imageResize
    resized_image = cv2.resize(cropped_image, imageResize)

#   resized image is appended to the img_data list
    img_data.append(resized_image)

    output.append([180 - data[2], data[3]])

savedData = [img_data, output]

warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)
    
# print('saving the data...')
# print('Please wait...')
 numpy.save('modelImageData.npy', savedData)
# print('saved successfully!!')