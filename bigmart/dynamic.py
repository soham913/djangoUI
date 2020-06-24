import pandas
import numpy
from datetime import datetime


def creator(df,data,attribute):
	table = pandas.pivot_table(df,index=data,values=attribute)
	tab2 = table.reset_index()
	print(tab2)
	xlist = []
	ylist = []
	for var in tab2.index.values:
		xlist.append(list(tab2[data][tab2.index == var])[0])
		ylist.append(list(tab2[attribute][tab2.index == var])[0])

	return xlist,ylist

def monthly(df,item,incoming_month,incoming_year,attribute):
	df['Sale_Date'] = pandas.to_datetime(df['Sale_Date'],dayfirst=True)
	df['year'] = df['Sale_Date'].dt.year
	df['month'] = df['Sale_Date'].dt.month
	want = df[['Item_Type','Sale_Date',attribute]][(df.Item_Type == item) & (df.year == int(incoming_year)) & (df.month == int(incoming_month))]
	want['Sale_Date'] = want['Sale_Date'].dt.date
	want['Sale_Date'] = want['Sale_Date'].astype(str)
	xl = list(want['Sale_Date'])
	yl = list(want[attribute])
	return xl,yl


def yearly(df,item,incoming_year,attribute):
	df['Sale_Date'] = pandas.to_datetime(df['Sale_Date'],dayfirst=True)
	df['year'] = df['Sale_Date'].dt.year
	want = df[['Item_Type','Sale_Date',attribute]][(df.Item_Type == item) & (df.year == int(incoming_year))]
	want['Sale_Date'] = want['Sale_Date'].dt.date
	want['Sale_Date'] = want['Sale_Date'].astype(str)
	want = want.sort_values(['Sale_Date'], axis=0)
	#return creator(want,'Sale_Date',attribute)
	xl = list(want['Sale_Date'])
	yl = list(want[attribute])
	return xl,yl


def alltime(df,item,attribute):
	df['Sale_Date'] = pandas.to_datetime(df['Sale_Date'], dayfirst=True)
	want = df[['Sale_Date',attribute]][df.Item_Type == item]
	want['Sale_Date'] = want['Sale_Date'].dt.date
	want['Sale_Date'] = want['Sale_Date'].astype(str)
	xl = list(want['Sale_Date'])
	yl = list(want[attribute])
	return xl, yl
