from dynamic import *
import matplotlib.pyplot as plt

df = pandas.read_csv('clean.csv')


sd = 'Meat,0,0,Item_MRP'

l = sd.split(',')

print(l)
if(l[1] != '0'):
    x,y = monthly(df,l[0],l[1],l[2],l[3])
elif(l[1] == '0' and l[2] != '0'):
     x,y = yearly(df,l[0],l[2],l[3])
     print(x)
     print(y)
else:
    x,y = alltime(df,l[0],l[3])
    print(x)
    print(y)
#
# elif(l[1] == 0 and l[2] == 0):
#     daily()