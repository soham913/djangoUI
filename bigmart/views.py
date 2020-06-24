from django.shortcuts import render
import pandas as pd
# from datetime import datetime
# Saglyat kami sales tya time span madhe
# Create your views here.
def dashboard(request):
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'],dayfirst=True)
    df['year'] = df['Sale_Date'].dt.year
    df['month'] = df['Sale_Date'].dt.month
    cdf = df['Item_Type']
    returned = cdf.value_counts()
    x_axis = list(returned.index)
    y_axis = list(returned.values)
    context = {'x_Item_Type': x_axis, 'y_Count_Item_Type': y_axis}

    ddf = df['Item_Fat_Content']
    returned = ddf.value_counts()
    x_axis = list(returned.index)
    y_axis = list(returned.values)
    context.update({'x_Item_Fat_Content': x_axis, 'y_Count_Item_Fat_Content': y_axis})

    tierWiseAvgSales = pd.pivot_table(df,columns=['Outlet_Location_Type'],values=['Item_Outlet_Sales'])
    xl = list(tierWiseAvgSales.columns.values)
    yl = []
    for var in xl:
        yl.append(tierWiseAvgSales[var][0])

    context.update({'x_Outlet_Location_Type':xl,'y_Item_Outlet_Sales':yl})

    cd = df['Item_Outlet_Sales']

    context.update({'Sales_sum':"{:,}".format(int(cd.sum(axis=0)))})


    #card of item types
    cd = df['Item_Type']
    context.update({'No_OF_Items':cd.nunique()})

    cd = df['Item_Supplier']
    context.update({'No_Of_Suppliers':cd.nunique()})

    cd = df['Outlet_Identifier']
    context.update({'outletCounts':cd.nunique()})

    cd = df[['Sale_Date','Item_Outlet_Sales']]

    d17 = {}
    d18 = {}

    for i in range(len(cd['Sale_Date'])):
        if df['year'][i] == 2017 :
            if df['month'][i] not in d17:
                d17[df['month'][i]] = df['Item_Outlet_Sales'][i]
            else:
                d17[df['month'][i]] += df['Item_Outlet_Sales'][i]

        elif df['year'][i] == 2018 :
            if df['month'][i] not in d18:
                d18[df['month'][i]] = df['Item_Outlet_Sales'][i]

            else:
                d18[df['month'][i]] += df['Item_Outlet_Sales'][i]

    freq17 = {}
    freq18 = {}

    for i in range(0,len(df['Sale_Date'])):
        if df['year'][i] == 2017 :
            if df['month'][i] not in freq17:
                freq17[df['month'][i]] = 1
            else:
                freq17[df['month'][i]] += 1

        elif df['year'][i] == 2018 :
            if df['month'][i] not in freq18:
                freq18[df['month'][i]] = 1
            else:
                freq18[df['month'][i]] += 1


    for var in d17:
        d17[var] = d17[var]/freq17[var]

    for var in d18:
        d18[var] = d18[var]/freq18[var]

    x17L = []
    y17L = []
    for var in d17:
        x17L.append(var)
        y17L.append(int(d17[var]))

    x18L = []
    y18L = []
    for var in d18:
        x18L.append(var)
        y18L.append(int(d18[var]))

    n = len(x17L)
    for i in range(n):
        for j in range(0, n-i-1):
            if x17L[j] > x17L[j+1] :
                x17L[j], x17L[j+1] = x17L[j+1], x17L[j]
                y17L[j], y17L[j+1] = y17L[j+1], y17L[j]

    n = len(x18L)
    for i in range(n):
        for j in range(0, n-i-1):
            if x18L[j] > x18L[j+1] :
                x18L[j], x18L[j+1] = x18L[j+1], x18L[j]
                y18L[j], y18L[j+1] = y18L[j+1], y18L[j]

    context.update({'y17L':y17L})
    context.update({'y18L':y18L})

    # return render(request,"index.html",context = context)

    table = pd.pivot_table(df,index='Outlet_Identifier',values='Item_Outlet_Sales', aggfunc='sum')
    table.sort_values(by=('Item_Outlet_Sales'), inplace=True)
    tab2 = table.reset_index()
    xlist = []
    ylist = []
    for var in tab2.index.values:
        xlist.append(list(tab2['Outlet_Identifier'][tab2.index == var])[0])
        ylist.append(list(tab2['Item_Outlet_Sales'][tab2.index == var])[0])

    context.update({'xOutlet' : xlist,'yOutletSales' : ylist})

    table = pd.pivot_table(df,index='Item_Supplier',values='Item_Outlet_Sales', aggfunc='sum')
    table.sort_values(by=('Item_Outlet_Sales'), inplace=True)
    tab2 = table.reset_index()
    xlist = []
    ylist = []
    for var in tab2.index.values:
        xlist.append(list(tab2['Item_Supplier'][tab2.index == var])[0])
        ylist.append(list(tab2['Item_Outlet_Sales'][tab2.index == var])[0])

    context.update({'xSupplier' : xlist,'ySupplierSales' : ylist})

    table = pd.pivot_table(df,index='Outlet_Location_Type',values='Item_Outlet_Sales', aggfunc='sum')
    table.sort_values(by=('Item_Outlet_Sales'), inplace=True)
    tab2 = table.reset_index()
    xlist = []
    ylist = []
    for var in tab2.index.values:
        xlist.append(list(tab2['Outlet_Location_Type'][tab2.index == var])[0])
        ylist.append(list(tab2['Item_Outlet_Sales'][tab2.index == var])[0])

    context.update({'xTier' : xlist,'yTierSales' : ylist})

    x = ['020-040','040-060','060-080','080-100','100-120','120-140','140-160','160-180','180-200','200-220','220-240','240-260','260-280']
    y = []
    for var in x:
        y.append(df['Item_Outlet_Sales'][(df.Item_MRP > int(var[0:3])) & (df.Item_MRP < int(var[4:7]))].sum())

    context.update({'ySales' : y})


    return render(request, 'dashboard.html', context = context)




def analysisProduct(request):
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")

    cdf = df['Item_Type']
    returned = list(cdf.unique())
    context = {'x_Item_Type': returned}


    tempDF = pd.pivot_table(df, index='Item_Type', values = 'Item_Visibility').reset_index()
    xlist = []
    ylist = []
    for var in tempDF.index.values:
        xlist.append(list(tempDF['Item_Type'][tempDF.index == var])[0])
        ylist.append(list(tempDF['Item_Visibility'][tempDF.index == var])[0])

    context.update({'xRadarList' : xlist,'yVisibility' : ylist})

    tempDF = pd.pivot_table(df, index='Item_Type', values = 'Item_Outlet_Sales').reset_index()
    xlist = []
    ylist = []
    for var in tempDF.index.values:
        xlist.append(list(tempDF['Item_Type'][tempDF.index == var])[0])
        ylist.append(list(tempDF['Item_Outlet_Sales'][tempDF.index == var])[0])

    context.update({'xHBarList' : xlist,'yHBarSales' : ylist})


    tab = pd.pivot_table(df,index='Item_Type',values=['Item_Weight','Item_Visibility','Item_Outlet_Sales']).reset_index()
    itemList = []
    weightList = []
    visiList = []
    salesList = []

    class Point:
        def __init__(self, x, y, r, label):
            self.label = label
            self.x = x
            self.y = y
            self.r = r

    p = []

    for var in tab.index.values:
        # itemList.append(list(tab['Item_Type'][tab.index == var])[0])
        # weightList.append(list(tab['Item_Weight'][tab.index == var])[0])
        # visiList.append(list(tab['Item_Visibility'][tab.index == var])[0])
        # salesList.append(list(tab['Item_Outlet_Sales'][tab.index == var])[0])
        p.append( Point( list(tab['Item_Outlet_Sales'][tab.index == var])[0], list(tab['Item_Visibility'][tab.index == var])[0], list(tab['Item_Weight'][tab.index == var])[0], list(tab['Item_Type'][tab.index == var])[0] ) )

    context.update({'points' : p})




    return render(request, 'analysisProduct.html', context = context)

def add(request) :
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'],dayfirst=True)
    df['year'] = df['Sale_Date'].dt.year
    df['month'] = df['Sale_Date'].dt.month

    def creator(df,data,attribute):
        table = pd.pivot_table(df,index=data,values=attribute)
        tab2 = table.reset_index()
        xlist = []
        ylist = []
        for var in tab2.index.values:
            xlist.append(list(tab2[data][tab2.index == var])[0])
            ylist.append(list(tab2[attribute][tab2.index == var])[0])

        return xlist,ylist


    def monthly(df,item,incoming_month,incoming_year,attribute):

        want = df[['Item_Type','Sale_Date',attribute]][(df.Item_Type == item) & (df.year == int(incoming_year)) & (df.month == int(incoming_month))]
        want['Sale_Date'] = want['Sale_Date'].dt.date
        want['Sale_Date'] = want['Sale_Date'].astype(str)
        return creator(want, 'Sale_Date', attribute)

    def yearly(df,item,incoming_year,attribute):
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'],dayfirst=True)
        want = df[['Item_Type','Sale_Date',attribute]][(df.Item_Type == item) & (df.year == int(incoming_year))]
        want['Sale_Date'] = want['Sale_Date'].dt.month
        want['Sale_Date'] = want['Sale_Date'].astype(int)
        want = want.sort_values(['Sale_Date'], axis=0)
        return creator(want,'Sale_Date',attribute)



    sd = request.POST["parameters"]

    print(sd)
    l = sd.split(',')

    if l[int((len(l)-1)/2)] == 'false':
        x = []
        y = []
        if(l[1] != '0'):
            x,y = monthly(df,l[0],l[1],l[2],l[3])
        elif(l[1] == '0' and l[2] != '0'):
            x,y = yearly(df,l[0],l[2],l[3])
            x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
        context = {'x1Data' : x, 'y1Data' : y}
    else:
        mid = l.index('true')
        if(l[1] != '0'):
            fx,fy = monthly(df,l[0],l[1],l[2],l[3])
            sx,sy = monthly(df,l[mid+1],l[mid+2],l[mid+3],l[mid+4])
        elif(l[1] == '0' and l[2] != '0'):
            fx,fy = yearly(df,l[0],l[2],l[3])
            sx,sy = yearly(df, l[mid+1], l[mid+3], l[mid+4])
            fx = sx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]


        context = {'x1Data' : fx, 'y1Data' : fy, 'x2Data' : sx, 'y2Data' : sy}

    context.update({'list' :l})


    tempDF = pd.pivot_table(df, index='Item_Type', values = 'Item_Visibility').reset_index()
    xlist = []
    ylist = []
    for var in tempDF.index.values:
        xlist.append(list(tempDF['Item_Type'][tempDF.index == var])[0])
        ylist.append(list(tempDF['Item_Visibility'][tempDF.index == var])[0])


    tempDF = pd.pivot_table(df, index='Item_Type', values = 'Item_Outlet_Sales').reset_index()
    xlist = []
    ylist = []
    for var in tempDF.index.values:
        xlist.append(list(tempDF['Item_Type'][tempDF.index == var])[0])
        ylist.append(list(tempDF['Item_Outlet_Sales'][tempDF.index == var])[0])

    context.update({'xHBarList' : xlist,'yHBarSales' : ylist})


    tab = pd.pivot_table(df,index='Item_Type',values=['Item_Weight','Item_Visibility','Item_Outlet_Sales']).reset_index()
    itemList = []
    weightList = []
    visiList = []
    salesList = []

    class Point:
        def __init__(self, x, y, r, label):
            self.label = label
            self.x = x
            self.y = y
            self.r = r

    p = []

    for var in tab.index.values:
        # itemList.append(list(tab['Item_Type'][tab.index == var])[0])
        # weightList.append(list(tab['Item_Weight'][tab.index == var])[0])
        # visiList.append(list(tab['Item_Visibility'][tab.index == var])[0])
        # salesList.append(list(tab['Item_Outlet_Sales'][tab.index == var])[0])
        p.append( Point( list(tab['Item_Outlet_Sales'][tab.index == var])[0], list(tab['Item_Visibility'][tab.index == var])[0], list(tab['Item_Weight'][tab.index == var])[0], list(tab['Item_Type'][tab.index == var])[0] ) )

    context.update({'points' : p})


    tempDF = pd.pivot_table(df, index='Item_Type', values = 'Item_Visibility').reset_index()
    xlist = []
    ylist = []
    for var in tempDF.index.values:
        xlist.append(list(tempDF['Item_Type'][tempDF.index == var])[0])
        ylist.append(list(tempDF['Item_Visibility'][tempDF.index == var])[0])

    context.update({'xRadarList' : xlist,'yVisibility' : ylist})

    return render(request, 'taw.html', context = context)




def analysisOutlet(request):
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")
    x = df['Outlet_Identifier'].unique()
    outletList = sorted(list(x))
    return render(request, 'analysisOutlet.html', {'outletList' : outletList})

def analysisOutlet2(request) :
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")

    outlet = request.POST["parameters"]
    a = df['Item_Type'][df.Outlet_Identifier == outlet]
    x = a.unique()

    context = {}
    context.update({'x_Item_Type' : x})
    context.update({'parameters' : outlet})

    return render(request, 'outletChartSelection.html', context = context)

def showOutletChart(request) :
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")

    def creator(df,data,attribute):
        table = pd.pivot_table(df,index=data,values=attribute)
        tab2 = table.reset_index()
        print(tab2)
        xlist = []
        ylist = []
        for var in tab2.index.values:
            xlist.append(list(tab2[data][tab2.index == var])[0])
            ylist.append(list(tab2[attribute][tab2.index == var])[0])

        return xlist,ylist


    def outletmonthly(df,outlet,item,incmonth,incyear,attribute):
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], dayfirst=True)
        df['year'] = df['Sale_Date'].dt.year
        df['month'] = df['Sale_Date'].dt.month
        want = df[['Sale_Date',attribute]][(df.Outlet_Identifier == outlet) & (df.Item_Type == item) & (df.year == int(incyear)) & (df.month == int(incmonth))]
        want['Sale_Date'] = want['Sale_Date'].dt.date
        want['Sale_Date'] = want['Sale_Date'].astype(str)
        return creator(want,'Sale_Date',attribute)

    def outletyearly(df,outlet,item,incoming_year,attribute):
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'],dayfirst=True)
        df['year'] = df['Sale_Date'].dt.year
        want = df[['Sale_Date',attribute]][(df.Item_Type == item) & (df.year == int(incoming_year)) & (df.Outlet_Identifier == outlet)]
        want['Sale_Date'] = want['Sale_Date'].dt.month
        want['Sale_Date'] = want['Sale_Date'].astype(int)
        want = want.sort_values('Sale_Date', axis=0)
        x,y =  creator(want,'Sale_Date',attribute)
        xl = []
        yl = []
        for i in range(1,13):
            if i in x:
                xl.append(x[x.index(i)])
                yl.append(y[x.index(i)])
            else:
                xl.append(i)
                yl.append(0)
        return xl,yl

    s =  request.POST["parameters"]
    l = s.split(',')

    context = {'parameters' : l}
    print(l)
    if(l[4] == 'false'):
        if(l[1] != '0'):
            x,y = outletmonthly(df,l[len(l)-1],l[0],l[1],l[2],l[3])
            context.update({'x1Data' : x, 'y1Data' : y })
        elif(l[1] == '0' and l[2] != '0'):
            x,y = outletyearly(df,l[len(l)-1],l[0],l[2],l[3])
            x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
            context.update({'x1Data' : x, 'y1Data' : y })

    else:
        mid = l.index('true')
        if(l[1] != '0'):
            fx,fy = outletmonthly(df,l[len(l)-1],l[0],l[1],l[2],l[3])
            sx,sy = outletmonthly(df,l[len(l)-1],l[mid+1],l[mid+2],l[mid+3],l[mid+4])
            context.update({'x1Data' : fx, 'y1Data' : fy })
            context.update({'x2Data' : sx, 'y2Data' : sy })

        elif(l[1] == '0' and l[2] != '0'):
            fx,fy = outletyearly(df,l[len(l)-1],l[0],l[2],l[3])
            sx,sy = outletyearly(df,l[len(l)-1], l[mid+1],l[mid+3],l[mid+4])
            sx = fx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
            context.update({'x1Data' : fx, 'y1Data' : fy })
            context.update({'x2Data' : sx, 'y2Data' : sy })


    return render(request, 'outletCharts.html', context=context)




def analysisSupplier(request):
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")
    x = df['Item_Supplier'].unique()
    supplierList = sorted(list(x))
    return render(request, 'analysisSupplier.html', {'supplierList' : supplierList})

def analysisSupplier2(request) :
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")

    supplier = request.POST["parameters"]
    a = df['Item_Type'][df.Item_Supplier == supplier]
    x = a.unique()

    context = {}
    context.update({'x_Item_Type' : x})
    context.update({'parameters' : supplier})

    return render(request, 'supplierChartSelection.html', context = context)

def showSupplierChart(request) :
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")

    def creator(df,data,attribute):
        table = pd.pivot_table(df,index=data,values=attribute)
        tab2 = table.reset_index()
        print(tab2)
        xlist = []
        ylist = []
        for var in tab2.index.values:
            xlist.append(list(tab2[data][tab2.index == var])[0])
            ylist.append(list(tab2[attribute][tab2.index == var])[0])

        return xlist,ylist


    def suppliermonthly(df,supplier,item,incmonth,incyear,attribute):
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], dayfirst=True)
        df['year'] = df['Sale_Date'].dt.year
        df['month'] = df['Sale_Date'].dt.month
        want = df[['Sale_Date',attribute]][(df.Item_Supplier == supplier) & (df.Item_Type == item) & (df.year == int(incyear)) & (df.month == int(incmonth))]
        want['Sale_Date'] = want['Sale_Date'].dt.date
        want['Sale_Date'] = want['Sale_Date'].astype(str)
        return creator(want,'Sale_Date',attribute)

    def supplieryearly(df,supplier,item,incoming_year,attribute):
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'],dayfirst=True)
        df['year'] = df['Sale_Date'].dt.year
        want = df[['Sale_Date',attribute]][(df.Item_Type == item) & (df.year == int(incoming_year)) & (df.Item_Supplier == supplier)]
        want['Sale_Date'] = want['Sale_Date'].dt.month
        want['Sale_Date'] = want['Sale_Date'].astype(int)
        want = want.sort_values('Sale_Date', axis=0)
        x,y =  creator(want,'Sale_Date',attribute)
        xl = []
        yl = []
        for i in range(1,13):
            if i in x:
                xl.append(x[x.index(i)])
                yl.append(y[x.index(i)])
            else:
                xl.append(i)
                yl.append(0)
        return xl,yl

    s =  request.POST["parameters"]
    l = s.split(',')

    context = {'parameters' : l}
    print(l)
    if(l[4] == 'false'):
        if(l[1] != '0'):
            x,y = suppliermonthly(df,l[len(l)-1],l[0],l[1],l[2],l[3])
            context.update({'x1Data' : x, 'y1Data' : y })
        elif(l[1] == '0' and l[2] != '0'):
            x,y = supplieryearly(df,l[len(l)-1],l[0],l[2],l[3])
            x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
            context.update({'x1Data' : x, 'y1Data' : y })

    else:
        mid = l.index('true')
        if(l[1] != '0'):
            fx,fy = suppliermonthly(df,l[len(l)-1],l[0],l[1],l[2],l[3])
            sx,sy = suppliermonthly(df,l[len(l)-1],l[mid+1],l[mid+2],l[mid+3],l[mid+4])
            context.update({'x1Data' : fx, 'y1Data' : fy })
            context.update({'x2Data' : sx, 'y2Data' : sy })

        elif(l[1] == '0' and l[2] != '0'):
            fx,fy = supplieryearly(df,l[len(l)-1],l[0],l[2],l[3])
            sx,sy = supplieryearly(df,l[len(l)-1], l[mid+1],l[mid+3],l[mid+4])
            sx = fx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
            context.update({'x1Data' : fx, 'y1Data' : fy })
            context.update({'x2Data' : sx, 'y2Data' : sy })


    return render(request, 'supplierCharts.html', context=context)





def salesPrediction(request) :
    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")

    fatList = sorted(list(df['Item_Fat_Content'].unique()))
    context = {'fatList' : fatList}

    itemsList = sorted(list(df['Item_Type'].unique()))
    context.update({'itemsList' : itemsList})

    outletSizeList = sorted(list(df['Outlet_Size'].unique()))
    context.update({'outletSizeList' : outletSizeList})

    outletLocationTypeList = sorted(list(df['Outlet_Location_Type'].unique()))
    context.update({'outletLocationTypeList' : outletLocationTypeList})

    outletTypeList = sorted(list(df['Outlet_Type'].unique()))
    context.update({'outletTypeList' : outletTypeList})

    outletEstablishmentYearList = sorted(list(df['Outlet_Establishment_Year'].unique()))
    context.update({'outletEstablishmentYearList' : outletEstablishmentYearList})

    return render(request, 'salesPrediction.html', context = context)

def predict(request) :
    s = request.POST['parameters']

    from numpy import sqrt, asarray
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error
    from random import randint

    context = {}

    def regr(lr,X_train,Y_train,X_test,Y_test):
        lr.fit(X_train,Y_train)
        Y_pred = lr.predict(X_train)
        rmse = sqrt(mean_squared_error(Y_train,Y_pred))

        return lr,rmse

    def encode(data,predict):
        le = LabelEncoder()
        sample = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
        for i in sample:
            data[i] = le.fit_transform(data[i])
        data = pd.get_dummies(data, columns=sample)
        return data

    def data(df,predict):
        train = df[predict]
        train = encode(train,predict)
        X = train.values
        Y = df['Item_Outlet_Sales'].values
        lr = LinearRegression(normalize = True)
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
        lr,rmse = regr(lr,X_train,Y_train,X_test,Y_test)
        return lr,rmse

    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")
    df['Outlet_Establishment_Year'] = df['Outlet_Establishment_Year']
    predict = df.columns.drop(['Item_Identifier','Item_Outlet_Sales','Outlet_Identifier','Item_Supplier','Sale_Date'])
    lr,rmse = data(df,predict)

    l = s.split(',')
    context.update({'parameters' : l})

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

    fat_sorted = sorted(list(df['Item_Fat_Content'].unique()))
    items_sorted = sorted(list(df['Item_Type'].unique()))
    outlet_size_sorted = sorted(list(df['Outlet_Size'].unique()))
    outlet_location_type_sorted = sorted(list(df['Outlet_Location_Type'].unique()))
    outlet_type_sorted = sorted(list(df['Outlet_Type'].unique()))


    # DataFrame Creation
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

    ar = asarray([l])
    pred = float(lr.predict(ar))

    actual = df['Item_Outlet_Sales'][(df.Item_Weight == d['Item_Weight']) & (df.Item_MRP == d['Item_MRP']) &  (df.Outlet_Establishment_Year == int(l[3])) & (df.Item_Fat_Content == fat_sorted[d['Item_Fat_Content']]) & (df.Item_Type == items_sorted[d['Item_Type']]) & (df.Outlet_Size == outlet_size_sorted[d['Outlet_Size']]) & (df.Outlet_Location_Type == outlet_location_type_sorted[d['Outlet_Location_Type']]) & (df.Outlet_Type == outlet_type_sorted[d['Outlet_Type']])]
    if actual.empty == True:
        actual_val = -1
    else:
        actual_val = float(actual.values)


    fatList = sorted(list(df['Item_Fat_Content'].unique()))
    context.update({'fatList' : fatList})

    itemsList = sorted(list(df['Item_Type'].unique()))
    context.update({'itemsList' : itemsList})

    outletSizeList = sorted(list(df['Outlet_Size'].unique()))
    context.update({'outletSizeList' : outletSizeList})

    outletLocationTypeList = sorted(list(df['Outlet_Location_Type'].unique()))
    context.update({'outletLocationTypeList' : outletLocationTypeList})

    outletTypeList = sorted(list(df['Outlet_Type'].unique()))
    context.update({'outletTypeList' : outletTypeList})

    outletEstablishmentYearList = sorted(list(df['Outlet_Establishment_Year'].unique()))
    context.update({'outletEstablishmentYearList' : outletEstablishmentYearList})

    context['RMSE'] = rmse
    context['Actual'] = actual_val

    # print(actual_val)
    context['Predicted'] = pred

    return render(request, 'predict.html', context = context)




def aboutUs(request):
    return render(request, 'aboutUs.html')




def debug(request):

    df = pd.read_csv("/home/sinisterstrike/Programs/Python/SDL/Project/dataSets/bigmart/clean.csv")
    cdf = df['Item_Type']
    returned = list(cdf.unique())
    context = {'x_Item_Type': returned}

    return render(request, 'debug.html', context = context)

def empty(request):
    return render(request, 'empty.html')