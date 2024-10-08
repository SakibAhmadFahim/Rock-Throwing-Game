import random
import numpy
import math
import tensorflow as tf
from tqdm import tqdm

max_v = 100
max_angle = 90

# from game
screenWidth = 800
screenHeight = 600

bar_height = 160
barHeight = 160
barWidth = 20
barPosX = (screenWidth // 2) - (barWidth // 2)
player1limitX = (0, barPosX - barWidth - 60)
player2limitX = (barPosX + barWidth, screenWidth - barWidth)

final_data = []

g = 9.8 # gravity

positions = []

pos1 = player1limitX[0]

step = 1

# a nested loop 

while pos1 <= player1limitX[1]:
    pos2 = player2limitX[0]
    while pos2 <= player2limitX[1]:
                
        positions.append([pos1, pos2])
        
        pos2 += step
    pos1 += step
count = 0

for pos1, pos2 in tqdm(positions): 
    #For each pair of positions (pos1 and pos2), the code calculates the target distance by subtracting pos1 from pos2. 
    target = pos2 - pos1

    bar_dist = barPosX - pos1
    
    minAngle = math.atan((bar_height ) / (bar_dist - bar_dist * bar_dist / target)) * 180 / math.pi

    maxAngle = 90 - 0.5 * math.asin(g * target / (max_v * max_v)) * 180 / math.pi
    
    #The angle is computed as the average of the minimum and maximum angles.
    angle = (minAngle + maxAngle) / 2
    
    radian_angle = angle * numpy.pi / 180 #convert degree to radian
    t = numpy.sqrt((2 * target * numpy.tan(radian_angle)) / g) # calculate time of AI. t = sqrt(2*x*tan(radian))
    velocity = target / (t * numpy.cos(radian_angle)) # calculate velocity of AI. v = x/(t*cos(radian))
    
    index = random.randint(0,count)
    
    #A new list containing the target, bar_dist, angle, and velocity is inserted into the final_data list at the randomly generated index.
    final_data.insert(index, [target, bar_dist, angle, velocity])
    count += 1

# saves the contents of the final_data list to a NumPy binary file named "modelData.npy"
numpy.save('modelData.npy', final_data)
    
        
    