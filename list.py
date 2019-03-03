list1 = [1,1,2,2,2,2,3,3,3,4]
list2 = [4,5,6,7,7,8,9,0,0,0]

list1.count(2) # count the number of element 2, print 4

[a*b for a,b in zip(list1,list2)] # compute the multiplication of two lists by elements