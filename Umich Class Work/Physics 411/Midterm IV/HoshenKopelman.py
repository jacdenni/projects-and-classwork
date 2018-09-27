
# coding: utf-8

# In[ ]:

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

def MakeLattice(L):
    A = np.zeros((L, L))
    return A

def FirstRow(A, A_Labeled, labelsIndex, LabelsList):
    L = A.shape[0]
    
    if A[0][0] == 0:
        A_Labeled[0][0] = 0.
    else:
        A_Labeled[0][0] = labelsIndex
        labelsIndex += 1
        LabelsList.append(labelsIndex)
    
    for j in range(1, L):
        if A[0][j] == 0:
            A_Labeled[0][j] = 0.
        else:
            if A[0][j - 1] == 0.:
                A_Labeled[0][j] = labelsIndex
                labelsIndex += 1
                LabelsList.append(labelsIndex)
            else:
                A_Labeled[0][j] = LabelsList[int(A_Labeled[0][j - 1])]
    
    return labelsIndex

def FirstColumn(A, A_Labeled, labelsIndex, LabelsList):
    L = A.shape[1]
    for i in range(1, L):
        if A[i][0] == 0:
            A_Labeled[i][0] = 0.
        else:
            if A[i - 1][0] == 0.:
                A_Labeled[i][0] = labelsIndex
                labelsIndex += 1
                LabelsList.append(labelsIndex)
            else:
                A_Labeled[i][0] = LabelsList[int(A_Labeled[i - 1][0])]
    return labelsIndex

def OccupiedNeighbors(A, i, j, A_Labeled, LabelsList):
    if (A[i - 1][j] != 0) & (A[i][j - 1] == 0):
        A_Labeled[i][j] = LabelsList[int(A_Labeled[i - 1][j])]
    elif (A[i - 1][j] == 0) & (A[i][j - 1] != 0):
        A_Labeled[i][j] = LabelsList[int(A_Labeled[i][j - 1])]
    else:
        if LabelsList[int(A_Labeled[i - 1][j])] == LabelsList[int(A_Labeled[i][j - 1])]:
            A_Labeled[i][j] = A_Labeled[i - 1][j]
        else:
            ConflictCase(A, i, j, A_Labeled, LabelsList)
    return A_Labeled

def ConflictCase(A, i, j, A_Labeled, LabelsList):
    bigger = int(np.amax([A_Labeled[i - 1][j], A_Labeled[i][j - 1]]))
    smaller = int(np.amin([A_Labeled[i - 1][j], A_Labeled[i][j - 1]]))
    LabelsList[bigger] = LabelsList[smaller]
    A_Labeled[i][j] = smaller
    return A_Labeled

def RelabelMatrix(A, LabelsList):
    L = np.shape(A)[0]
    for i in range(L):
        for j in range(L):
            A[i][j] = LabelsList[int(A[i][j])]
    return np.float32(A)

def RelabelList(myList):
    for i in range(1, len(myList)):
        if (myList[i] - myList[i - 1]) <= 0.:
            for j in range(i + 1, len(myList)):
                if myList.index(myList[j]) == j:
                    myList[j] -= 1.
    return myList

def HoshenKopelman(A):
    L = A.shape[0]
    A_Labeled = np.zeros((L, L))
    LabelsList = [0, 1]
    labelsIndex = 1
    
    labelsIndex = FirstRow(A, A_Labeled, labelsIndex, LabelsList)
    labelsIndex = FirstColumn(A, A_Labeled, labelsIndex, LabelsList)
    for i in range(1, L):
        for j in range(1, L):
            if A[i][j] == 0.:
                A_Labeled[i][j] = 0
            elif (A[i - 1][j] == 0) & (A[i][j - 1] == 0):
                A_Labeled[i][j] = labelsIndex
                labelsIndex += 1
                LabelsList.append(labelsIndex)
            else:
                OccupiedNeighbors(A, i, j, A_Labeled, LabelsList)
    
    RelabelList(LabelsList)
    RelabelMatrix(A_Labeled, LabelsList)
    
    return A_Labeled

def FillPointsInLattice(A, points):
    for point in points:
        A[point] = 1.
    return A

def PlotLabeledLattice(A_Labeled, filename):
    L = A_Labeled.shape[0]
    x, y = np.mgrid[slice(0, L + 1, 1), slice(0, L + 1, 1)]
    plt.figure(0)
    plt.clf()
    plt.pcolor(x, y, A_Labeled, cmap = 'Paired', vmin = 0., vmax = np.amax(A_Labeled))
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.savefig(filename)
    return None

def OccupyALattice(L, p):
    A = MakeLattice(L)
    for i in range(L):
        for j in range(L):
            if rand.random() > p:
                A[i][j] = 0
            else:
                A[i][j] = 1
    return A

