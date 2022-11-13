# sf
#aim == write the code to calculate net input and apply activation function

import numpy as np

def sig(x):

return 1/(1+ np.exp(-x))
bias=float(input("enter the value of bias"))

#n will take neurons of the network.

n=int(input("enter the number of input neurons:"))

#w will take the weight & x will take the input

w=[]

x=[]

#taking the value of input and their weight

for i in range(0,n):

a=float(input("enter the input: "))

x.append(a)

b=float(input("enter the weights: "))

w.append(b)

print("the given weights are: ")

print(w)

print("the given inputs are: ")

print(x)

y=bias

for i in range(0,n):

y=y+(w[i]*x[i])

print("the calculated net input y:")

print(y)

binary=sig(y)

print("the output after applying binary sigmoidal function activation")

print(round(binary,3))

2A

num_ip=int(input("enter the number of inputs"))

w1=1

w2=1

print("for the",num_ip,"inputs calculate the net input using yin=x1w1+x2w2")

x1=[]

x2=[]

for j in range(0,num_ip):

ele1= int(input("x1= "))

ele2 =int(input("x2= "))

x1.append(ele1)

x2.append(ele2)

print("x1= " , x1)

print("x2+ ", x2)

n=x1*w1

m=x2*w2

Yin=[]

for i in range(0,num_ip):

Yin.append(n[i]+m[i])

print("Yin= ",Yin)

Yin = []

for i in range(0,num_ip):

Yin.append(n[i]-m[i])

print("after assuming one weight as excitatory and the other as inhibitory Yin", Yin)

Y=[]

for i in range(0,num_ip):

if(Yin[i]>=1):

ele =1

Y.append(ele)

if(Yin[i]<1):

ele=0

Y.append(ele)

print("Y=", Y)

2B

#Getting weights and threshold value

print('Enter weights');

w11=int(input('Weight w11='));

w12=int(input('weight w12='));

w21=int(input('Weight w21='));

w22=int(input('weight w22='));

v1=int(input('weight v1='));

v2=int(input('weight v2='));

print('Enter Threshold Value');

theta=int(input('theta='));

x1=[0,0,1,1];

x2=[0,1,0,1];

z=[0,1,1,0];

con=1;

zin1=[0,0,0,0]

zin2=[0,0,0,0]

y1=[0,0,0,0]

y2=[0,0,0,0]

yin=[0,0,0,0]

y=[0,0,0,0]

while con:

for i in range(0,3):

zin1[i]=x1[i]*w11+x2[i]*w21;

zin2[i]=x1[i]*w21+x2[i]*w22;

for i in range(0,3):

if zin1[i]>=theta:

y1[i]=1;

else:

y1[i]=0;

if zin2[i]>=theta:

y2[i]=1

else:

y2[i]=0

for i in range(0,3):

yin[i]=y1[i]*v1+y2[i]*v2

print(yin)

for i in range(0,3):

if yin[i]>=theta:

y[i]=1;

else:

y[i]=0;

print('Output of Net');

print(y);

if y==z:

con=0;

else:

print('Net is not learning enter another set of weights and Threshold value');

w11=input('Weight w11=');

w12=input('weight w12=');

w21=input('Weight w21=');

w22=input('weight w22=');

v1=input('weight v1=');

v2=input('weight v2=');

theta=input('theta=');

#endwhile

print('McCulloch-Pitts Net for XOR function');

print('Weights of Neuron Z1');

print(w11);

print(w21);

print('weights of Neuron Z2');

print(w12);

print(w22);

print('weights of Neuron Y');

print(v1);

print(v2);

print('Threshold value');

print(theta);

10A GENETIC ALGORITHM

#following statement will create an empty two dimentional array to store offspring
offspring = numpy.empty((offspringSize, population.shape[1]))

for k in range (offspringSize):

#determining the crossover point

crossoverPoint = numpy.random.randint(0,genes)

#index of the first parent

parent1Index = k%parents.shape[0]

    #index of the second
parent2Index = (k+1) %parents.shape[0]

#extracting first half of the offspring

offspring[k,0: crossoverPoint]= parents[parent1Index,0:crossoverPoint]

#extracting second half of the offspring

offspring[k, crossoverPoint:] = parents[parent2Index, crossoverPoint:]

print("\n offspring after crossover:")

print(offspring)

#implementation of random initialization mutation.

for index in range(offspring.shape[0]):

randomIndex = numpy.random.randint(1,genes)

randomValue=numpy.random.uniform(lb,ub,1)

offspring[index,randomIndex] = offspring[index, randomIndex]+ randomValue

print("\n new population for next generation:")

print(population)

fitness=numpy.sum(population*population, axis=1)

fittestIndex=numpy.where(fitness == numpy.max(fitness))

#extracting index of fittest chromosome

fittestIndex = fittestIndex[0][0]

#getting best chromosome

fittestInd= population[fittestIndex, :]

bestFitness= fitness[fittestIndex]

print("\n best individual")

print(fittestInd)

print("\n best individual fitness:")

print(bestFitness)

10B GENETIC ALGORITHM

import numpy

#parameter initialization

genes=2

chromosomes=10

mattingPoolSize=6

offspringSize= chromosomes - mattingPoolSize

lb=-5

ub=5

populationSize = (chromosomes, genes)

generations = 3

#Population initialization

population = numpy.random.uniform(lb, ub, populationSize)

for generation in range(generations):

print(("Generation:", generation+1))

fitness= numpy.sum(population*population, axis=1)

print("\n population")

print(population)

print("\nfitness calculation")

print(fitness)

#following statement will create an empty two dimensional array to store parents
parents= numpy.empty((mattingPoolSize, population.shape[1]))

#a loop to extract one parent in each iteration

for p in range(mattingPoolSize):

#finding index of fittest chromosome in the population

fittestIndex=numpy.where(fitness == numpy.max(fitness))

#extracting index of fittest chromosome

fittestIndex = fittestIndex[0][0]

#coping fittest chromosome into parents array

parents[p,:] = population[fittestIndex, :]

#changing fitness of fittest chromosome to avoid reselction of that chromosome

fitness[fittestIndex]=-1

print("\n Parents:")

print(parents)

#following statement will create an empty two dimentional array to store offspring

offspring = numpy.empty((offspringSize, population.shape[1]))

for k in range (offspringSize):

#determining the crossover point

crossoverPoint = numpy.random.randint(0,genes)

#index of the first parent

parent1Index = k%parents.shape[0]

#index of the second

parent2Index = (k+1) %parents.shape[0]

#extracting first half of the offspring

offspring[k,0: crossoverPoint]= parents[parent1Index,0:crossoverPoint]

#extracting second half of the offspring

offspring[k, crossoverPoint:] = parents[parent2Index, crossoverPoint:]

print("\n offspring after crossover:")

print(offspring)

#implementation of random initialization mutation.

for index in range(offspring.shape[0]):

randomIndex = numpy.random.randint(1,genes)

randomValue=numpy.random.uniform(lb,ub,1)

offspring[index,randomIndex] = offspring[index, randomIndex]+ randomValue

print("\n new population for next generation:")

print(population)

fitness=numpy.sum(population*population, axis=1)

fittestIndex=numpy.where(fitness == numpy.max(fitness))

#extracting index of fittest chromosome

fittestIndex = fittestIndex[0][0]

#getting best chromosome

fittestInd= population[fittestIndex, :]

bestFitness= fitness[fittestIndex]

print("\n best individual")

print(fittestInd)

print("\n best individual fitness:")

print(bestFitness)

SIGMOD

import numpy as np

import matplotlib.pyplot as plt

def sig(x):

return 1/(1 + np.exp(-x))

x = 1.0

print('Applying Sigmoid Activation on (%.lf) gives %.lf' % (x, sig(x)))

x= -10.0

print('Applying Sigmoid Activation on (%.lf) gives %.lf' % (x, sig(x)))

x= 0.0

print('Applying Sigmoid Activation on (%.lf) gives %.lf' % (x, sig(x)))

x= 15.0

print('Applying Sigmoid Activation on (%.lf) gives %.lf' % (x, sig(x)))

x= -2.0

print('Applying Sigmoid Activation on (%.lf) gives %.lf' % (x, sig(x)))

x = np.linspace(-10,10,50)

p = sig(x)

plt.xlabel("x")

plt.ylabel("Sigmoid(x)")

plt.plot(x, p)

plt.show()

plt.show()

FUZZY SET A

A=[0.5,0.1,0.63,0.2]

B=[0.2,0.4,0.2,0.1]

Union=[]

Intersection=[]

print("fuzzy set A=",A)

print("fuzzy set B= ", B)

for i in range(0,4):

if(A[i]>B[i]):

Union.append(A[i])

else:

Union.append(B[i])

print("union set", Union)

for i in range(0,4):

if(A[i]>B[i]):

Intersection.append(A[i])

else:

Intersection.append(B[i])

print("Intersection set", Intersection)

CA=[]

print("complement for fuzzy set A ")

for i in range(0,4):

CA.append(1-A[i])

print("complement set=" ,CA)

CB=[]

print("complement for fuzzy set B ")

for i in range(0,4):

CB.append(1-B[i])

print("complement set=" ,CB)

P=[]

print("product of fuzzy set a and b")

for i in range(0,4):

P.append(round(A[i]+ B[i],2))

print("product of the fuzzy set A and B",P)

sum=[]

for i in range(0,4):

sum.append(round(A[i]+B[i],2))

print("sum of fuzzy set A and B",sum)

BU=[]

for i in range(0,4):

if(sum[i]>1):

BU.append(1)

else:

BU.append(sum[1])

FUZZY SET B

A=[0.5,0.1,0.63,0.2]

B=[0.2,0.4,0.2,0.1]

Union=[]

Intersection=[]

print("fuzzy set A=",A) print("fuzzy set B= ", B)

for i in range(0,4):

if(A[i]>B[i]):

Union.append(A[i])

else:

Union.append(B[i])

print("union set", Union)

for i in range(0,4):

if(A[i]>B[i]):

Intersection.append(A[i])

else:

Intersection.append(B[i])

print("Intersection set", Intersection)

CA=[]

print("complement for fuzzy set A ")

for i in range(0,4):

CA.append(1-A[i])

print("complement set=" ,CA)

CB=[]

print("complement for fuzzy set B ")

for i in range(0,4):

CB.append(1-B[i])

print("complement set=" ,CB)

P=[]

print("product of fuzzy set a and b")

for i in range(0,4):

P.append(round(A[i]+ B[i],2))

print("product of the fuzzy set A and B",P)

sum=[]

for i in range(0,4):

sum.append(round(A[i]+B[i],2))

print("sum of fuzzy set A and B",sum
