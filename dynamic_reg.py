import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import numpy

def regr(lr,X_train,Y_train,X_test,Y_test):
    lr.fit(X_train,Y_train)
    Y_pred = lr.predict(X_test)
    print('RMSE : ',numpy.sqrt(mean_squared_error(Y_test,Y_pred)))
    return lr


#used for converting categorical data into usable Numerical data
def encode(data,predict):
    #Label Encoding preliminary step for one hot encoding
    le = LabelEncoder()
    sample = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
    #var = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type']
    #var = []
    #for i in predict:
    #	if i in sample:
    #		var.append(i)
    for i in sample:
        data[i] = le.fit_transform(data[i])

    #One Hot Encoding using get_dummies()
    data = pandas.get_dummies(data, columns=sample)
    #data.to_csv('param.csv',index=False)
    return data







def data(df,predict,d):
	train = df[predict]
	train = encode(train,predict)
	#print(train.columns.values)
	#print(train.head())
	X = train.values
	Y = df['Item_Outlet_Sales'].values

	lr = LinearRegression()
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

	lr = regr(lr,X_train,Y_train,X_test,Y_test)
	#items_sorted = sorted(list(df['Item_Type'].unique()))
	#DataFrame Creation 
	features = ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
	l = []
	for var in features:
		if var == 'Item_Fat_Content':
			for i in range(0,2):
				if i == d[var]:
					l.append(1)
				else:
					l.append(0)
		elif var == 'Item_Type':
			for i in range(0,16):
				if i == d[var]:
					l.append(1)
				else:
					l.append(0)	
		elif var == 'Outlet_Size':
			for i in range(0,3):
				if i == d[var]:
					l.append(1)
				else:
					l.append(0)
		elif var == 'Outlet_Location_Type':
			for i in range(0,3):
				if i == d[var]:
					l.append(1)
				else:
					l.append(0)
		elif var == 'Outlet_Type':
			for i in range(0,4):
				if i == d[var]:
					l.append(1)
				else:
					l.append(0)		
		else:
			l.append(d[var])

	print(l)

	ar = numpy.asarray([l])
	return lr.predict(ar)










df = pandas.read_csv('clean.csv')
predict = df.columns.drop(['Item_Identifier','Item_Outlet_Sales','Outlet_Identifier','Item_Supplier','Sale_Date'])

s = '9.3,249.8092,0.01,1999,0,4,1,0,1'

l = s.split(',')

print(l)

d={
	'Item_Weight' : float(l[0]),
	'Item_MRP' : float(l[1]),
	'Item_Visibility' : float(l[2]),
	'Outlet_Establishment_Year' : int(l[3]),
	'Item_Fat_Content' : int(l[4]),
	'Item_Type': int(l[5]),
	'Outlet_Size' : int(l[6]),
	'Outlet_Location_Type':int(l[7]),
	'Outlet_Type': int(l[8])
}


#print(d)

fat_sorted = sorted(list(df['Item_Fat_Content'].unique()))
items_sorted = sorted(list(df['Item_Type'].unique()))
outlet_size_sorted = sorted(list(df['Outlet_Size'].unique()))
outlet_location_type_sorted = sorted(list(df['Outlet_Location_Type'].unique()))
outlet_type_sorted = sorted(list(df['Outlet_Type'].unique()))
# for var in d:
# 	if (var == 'Item_Fat_Content'):
# 		d[var] = fat_sorted.index(d[var])
# 	elif (var == 'Item_Type'):
# 		d[var] = items_sorted.index(d[var])
# 	elif (var == 'Outlet_Size'):
# 		d[var] = outlet_size_sorted.index(d[var])
# 	elif (var == 'Outlet_Location_Type'):
# 		d[var] = outlet_location_type_sorted.index(d[var])
# 	elif (var == 'Outlet_Type'):
# 		d[var] = outlet_type_sorted.index(d[var])


# print(d)

pred = data(df,predict,d)
print('Predicted ',float(pred))

actual = df['Item_Outlet_Sales'][(df.Item_Weight == d['Item_Weight']) & (df.Item_MRP == d['Item_MRP']) &  (df.Outlet_Establishment_Year == d['Outlet_Establishment_Year']) & (df.Item_Fat_Content == fat_sorted[d['Item_Fat_Content']]) & (df.Item_Type == items_sorted[d['Item_Type']]) & (df.Outlet_Size == outlet_size_sorted[d['Outlet_Size']]) & (df.Outlet_Location_Type == outlet_location_type_sorted[d['Outlet_Location_Type']]) & (df.Outlet_Type == outlet_type_sorted[d['Outlet_Type']])]
if actual.empty == True:
	print('empty')
else:
	print('Actual ',float(actual.values))	
#print('Actual ',(actual.values))
#(df.Item_Visibility == d['Item_Visibility']) &
# print(df['Item_Outlet_Sales'][(df.Item_Weight == 9.3) & (df.Item_MRP == 249.8092) & (df.Item_Visibility == 0.016047)])