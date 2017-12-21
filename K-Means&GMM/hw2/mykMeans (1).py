import pandas
import numpy as np
import pylab as pl
import math
import copy

datalist=[]
c1=[]
c2=[]
c3=[]

#Calculating distance between two points in 2 dimensional space
def distance2pts(x1, y1, x2, y2):
    x_value = abs( x1 - x2 )
    x_value = pow(x_value,2)
    y_value = abs( y1 - y2 )
    y_value = pow( y_value, 2 )
    sum = x_value + y_value
    act_dist = pow( sum, 0.5 )
    return act_dist

#Calculating Mean of a list belonging to a centroid
def mean(cluster_list):
    x_sum = 0
    y_sum = 0
    mean_coord = []
    for i in cluster_list:
       x_sum+=datalist[0][i]
       y_sum+=datalist[1][i]
    count=len(cluster_list)
    x_coord=x_sum/count
    mean_coord.append(x_coord)
    y_coord=y_sum/count
    mean_coord.append(y_coord)
    return mean_coord

def kmeans_algo(centroidlist):
    centroidlist1 = []
    centroidlist2 = []
    centroidlist3 = []
    final=[[]]*2
    meanC1=[]
    meanC2 = []
    meanC3 = []
    for i in range (0,len(datalist[0]),1):
        d1=distance2pts(centroidlist[0][0],centroidlist[1][0],datalist[0][i],datalist[1][i])
        d2=distance2pts(centroidlist[0][1],centroidlist[1][1],datalist[0][i],datalist[1][i])
        d3=distance2pts(centroidlist[0][2],centroidlist[1][2],datalist[0][i],datalist[1][i])
        min_d123 = min( d1, d2, d3 )
        if min_d123 == d1:
            centroidlist1.append(i)
        elif min_d123 == d2:
            centroidlist2.append(i)
        else:
            centroidlist3.append(i)
    graph(centroidlist1,'bo')       #dots in graph of centroidlist1
    graph(centroidlist2,'go')       #dots in graph of centroidlist2
    graph(centroidlist3,'yo')       #dots in graph of centroidlist3
    meanC1 = mean( centroidlist1 )
    meanC2 = mean( centroidlist2 )
    meanC3 = mean( centroidlist3 )
    final[0]=meanC1[0],meanC2[0],meanC3[0]
    final[1]=meanC1[1],meanC2[1],meanC3[1]
    return final

def graph(cluster_list,color):
    x=[]
    y=[]
    for i in cluster_list:
        x.append(datalist[0][i])
        y.append(datalist[1][i])
    pl.plot( x, y,color)


if __name__ == "__main__":
    dataframe = pandas.read_csv('clusters.txt', header=None)
    centroidlist = (dataframe.sample(n=3)).values.T.tolist()
    datalist =dataframe.values.T.tolist()
    print centroidlist
    while True:
        centroidlist_copy = copy.deepcopy(centroidlist)
        centroidlist=kmeans_algo(centroidlist)
        if centroidlist_copy == centroidlist :
            break
    print 'Final 3 Centroids of K-means Algorithm:'
    print '[',centroidlist[0][0],',',centroidlist[1][0],']'
    print '[',centroidlist[0][1],',',centroidlist[1][1],']'
    print '[',centroidlist[0][2],',',centroidlist[1][2],']'

    pl.plot(centroidlist[0][0], centroidlist[1][0],'r*')    #Red color Star for Centroids
    pl.plot(centroidlist[0][1], centroidlist[1][1],'r*')    #Red color Star for Centroids
    pl.plot(centroidlist[0][2], centroidlist[1][2],'r*')    #Red color Star for Centroids
    pl.xlabel('X Axis')
    pl.ylabel('Y Axis')
    pl.show()                                               # Show the graph