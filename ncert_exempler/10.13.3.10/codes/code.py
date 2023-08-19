import numpy as np
import matplotlib.pyplot as plt

#Number of samples is 10
simlen=int(10)

#choosing random number from 1 to 10
num = np.random.randint(1,11, size=simlen)

#choosing random suit
suits = ['Diamonds', 'Hearts', 'Spades', 'Clubs']
shape = np.random.choice(suits,size=simlen)

#showing the new cases generated
ls = eq = gr = 0
print("Random cases generated:")
for i in range(0,10):
    print(num[i]," - ",shape[i])
    if(num[i]<7):
        ls += 1
    elif(num[i]==7):
        eq += 1
    else:
        gr += 1

#printing probabilities
print("\nProbability of:")
print("Less than 7 - simulation, actual : ",ls/simlen,", 0.6")
print("Equal to 7 - simulation, actual : ",eq/simlen,", 0.1")
print("Greater than 7 - simulation, actual : ",gr/simlen,", 0.3")
