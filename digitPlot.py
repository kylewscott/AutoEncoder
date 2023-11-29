import matplotlib.pyplot as plt
import numpy as np
import csv

############################################################
# PLots the digit found inisde the digit.csv file which is #
# created using the plotDigitInput function in mnist.cpp   #
############################################################

data = []
with open('digit.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        for i in range(len(row)-1):
            data.append(float(row[i]))

#Reshape into 28x28 matrix
datax = np.reshape(data, (28,28))

#Plot digit
plt.figure(figsize=(4, 4))
plt.imshow(datax, cmap='gray')
plt.axis('off')
plt.show()
