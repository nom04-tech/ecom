import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from streamlit import caching
import time
import datetime as dt
import base64
import plotly.graph_objects as go


st.set_page_config(layout="wide")



st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#F5F5F5,#F5F5F5);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)
st.write ("""<style> body {
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAf89RUWKjFNjFMUzDCn9LZjlxVF_Iz871Dw&usqp=CAU");
  background-repeat: repeat;


} </style>""", unsafe_allow_html=True)
st.write(""" <style>body {
  margin: 40px;
}

.box {
  background-color: #444;
  color: #fff;
  border-radius: 5px;
  padding: 20px;
  font-size: 150%;
}

.box:nth-child(even) {
  background-color: #ccc;
  color: #000;
}

.wrapper {
  width: 600px;
  display: grid;
  grid-gap: 10px;
  grid-template-columns: repeat(6, 100px);
  grid-template-rows: 100px 100px 100px;
  grid-auto-flow: column;
}</style>""",unsafe_allow_html=True)
## to hide the streamlit text in the bottom of the page
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#######################################################################

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from streamlit import caching
import time
import datetime as dt
import base64
data=pd.read_csv('superstore.csv', encoding= 'unicode_escape')
data['Order Date']= pd.to_datetime(data['Order Date'])
st.sidebar.markdown("<h1> Table of Contents",unsafe_allow_html=True)
navigation=st.sidebar.radio('Navigate pages',['Data Overview','Exploratory Data Analysis','Predictive Analysis'])
if navigation=='Data Overview':
    st.title("""Data Information """)
    st.markdown("""<br><div>The dataset contains the following columns:</div><br>""",unsafe_allow_html=True)
    st.markdown("""<div role="tabpanel" aria-labelledby="tab-title-data_fields_-tab" style="display: block;"><div><div
    ><div><table><tbody><tr><th>Name</th><th>Description</th></tr><tr>
    <td>Order_ID</td><td>Identifies the unique ID of the order</td>
    <tr><td>Orderdate</td><td>Identifies the Order date</td></tr>
    <tr><td>Ship Mode</td><td>Identifies the type of Shipment ex First Class, Same Day, Second Class, Standard Class</td></tr>
    <tr><td>Customer_ID</td><td>Identifies the unique ID of the order</td></tr>
    <tr><td>Segment</td><td>Identifies the type of the Segment Consumer Corporate or Home office</td></tr>
    <tr><td>City</td><td>Identifies the City where the order was placed</td></tr>
    <tr><td>Country</td><td>Identifies the Country where the order was placed</td></tr>
    <tr><td>Region</td><td>Identifies the Region where the order was placed</td></tr>
    <tr><td>ProductID</td><td>Identifies the unique ID of the Product</td></tr>
    <tr><td>Category</td><td>Identifies the Category of the product</td></tr>
    <tr><td>Product_Name</td><td>Identifies the Product Name </td></tr>
    <tr><td>Sales</td><td>The amount of Sales generated from the order</td></tr>
    <tr><td>Quantity</td><td>The amount of units ordered</td></tr>
    <tr><td>Discount</td><td>The discount of this product</td></tr>
    <tr><td>Profit</td><td>The profit generated from this order</td></tr>
    <tr><td>Shipping Cost</td><td>The shipping cost of this order</tr>
    <tr><td>Order Priority</td><td>The priority is categorized between Critical High Medium Low</td></tr>
    <tr><td>Quantity</td><td>The amount of units ordered</td></tr>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2=st.beta_columns(2)

 
    Filtered_data=col1.multiselect("Filter the data you want by selecting the columns you want",data.columns.tolist())
    col1.write(data.filter(Filtered_data).head(10))
    a= col2.checkbox('Check the type of variables found in the data')
    if a:
        col2.write(data.dtypes)
    # st.markdown("<br>",unsafe_allow_html=True)
    col1,col2,col3,col4=st.beta_columns((1,2,2,2))

if navigation=='Exploratory Data Analysis':
    st.title("""Exploratory Data Analysis""")
    st.header("""Variation of Profit with time""")
    # data
    col1,col2=st.beta_columns(2)
    time_profit=data.set_index('Order Date').groupby(pd.Grouper(freq='M'))['Profit'].sum().reset_index()
    maxvalue = time_profit['Profit'].max()
    minvalue = time_profit['Profit'].min()
    max_year=time_profit['Order Date'][time_profit['Profit']==maxvalue].values[0]
    min_year=time_profit['Order Date'][time_profit['Profit']==minvalue].values[0]
    
    col1,col2=st.beta_columns(2)
    col1.markdown("<b> The highest Profit value was in {} to reach {} $ </b>".format(pd.to_datetime(str(max_year)).strftime('%Y.%m.%d'),round(maxvalue,2)),unsafe_allow_html=True)
    col1.markdown("<b> The lowest Profit value was in {} to reach {} $ </b>".format(pd.to_datetime(str(min_year)).strftime('%Y.%m.%d'),round(minvalue,2)),unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    fig1 = go.Figure(go.Scatter(x=time_profit['Order Date'], y=time_profit.Profit, mode='lines+markers+text'))
    fig1.update_layout(
    title="The variation of Profit over Time",
    xaxis_title="Time",
    yaxis_title="Profit $",
  
    font=dict(
        family="Calibri",
        size=18,
        color="RebeccaPurple"
    )
)


    col1.plotly_chart(fig1)
    time_Orders=data.set_index('Order Date').groupby(pd.Grouper(freq='M'))['Order ID'].count().reset_index()
    fig2 = go.Figure(go.Scatter(x=time_Orders['Order Date'], y=time_Orders['Order ID'], mode='lines+markers+text'))
    fig2.update_layout(
    title="The variation of Orders over Time",
    xaxis_title="Time",
    yaxis_title="Number of Orders",
    legend_title="Legend Title",
    font=dict(
        family="Calibri",
        size=18,
        color="RebeccaPurple"
    )
)

    col2.plotly_chart(fig2)
    maxvalue_ort = time_Orders['Order ID'].max()
    minvalue_ort = time_Orders['Order ID'].min()
    max_year_ort=time_Orders['Order Date'][time_Orders['Order ID']==maxvalue_ort].values[0]
    min_year_ort=time_Orders['Order Date'][time_Orders['Order ID']==minvalue_ort].values[0]
    
    col2.markdown("<b>The highest ordering was in {} to reach {} $ </b>".format(pd.to_datetime(str(max_year_ort)).strftime('%Y.%m.%d'),round(maxvalue_ort,2)),unsafe_allow_html=True)
    col2.markdown("<b>The lowest ordering was in {} to reach {} $ </b>".format(pd.to_datetime(str(min_year_ort)).strftime('%Y.%m.%d'),round(minvalue_ort,2)),unsafe_allow_html=True)

    st.markdown("""<br>""",unsafe_allow_html=True)
    col1,col2,col3,col4,col5=st.beta_columns((1,1,2,2,1))
    cust_order=data.set_index('Order Date').groupby([pd.Grouper(freq='M'),'Country'])['Order ID'].count().reset_index()
    Country_filter = col1.multiselect('Select the Country ',cust_order['Country'].unique())

    if  Country_filter!= []:
        cust_order_1=cust_order.loc[(cust_order['Country'].isin(Country_filter))]
        maxvalue_order = cust_order_1['Order ID'].max()
        minvalue_order = cust_order_1['Order ID'].min()
        max_year_order=cust_order_1['Order Date'][cust_order_1['Order ID']==maxvalue_order].values[0]
        min_year_order=cust_order_1['Order Date'][cust_order_1['Order ID']==minvalue_order].values[0]

        fig2 = px.line(cust_order_1,x='Order Date', y='Order ID',color='Country',line_group="Country",width=500,height=500)
        fig2.update_layout(
    title="The variation of Orders over Time with respect <br> to countries Selected",

    font=dict(
        family="Calibri",
        size=14,
        color="RebeccaPurple"
    )
)
        col2.plotly_chart(fig2)



    else:
        col1.markdown("<h4 style='text-align:center; color: black;'> Please select a Country",unsafe_allow_html=True)

    Shippingcost_order=data.set_index('Order Date').groupby([pd.Grouper(freq='M'),'Country'])['Shipping Cost'].count().reset_index()

    if  Country_filter!= []:
        Shippingcost_order_1=Shippingcost_order.loc[(cust_order['Country'].isin(Country_filter))]
        maxvalue_cost = Shippingcost_order_1['Shipping Cost'].max()
        minvalue_cost = Shippingcost_order_1['Shipping Cost'].min()
        max_year_cost=Shippingcost_order_1['Order Date'][Shippingcost_order_1['Shipping Cost']==maxvalue_cost].values[0]
        min_year_cost=Shippingcost_order_1['Order Date'][Shippingcost_order_1['Shipping Cost']==minvalue_cost].values[0]

        fig2 = px.line(Shippingcost_order_1,x='Order Date', y='Shipping Cost',color='Country',line_group="Country",width=500,height=500)
        fig2.update_layout(
    title="""The variation of Shipping cost over Time with respect <br> to countries Selected""",

    font=dict(
        family="Calibri",
        size=14,
        color="RebeccaPurple"
    )
)
        col4.plotly_chart(fig2)
        col1,col2,col3,col4,col5=st.beta_columns((1,2,1,2,1))

        col2.markdown("<b>In {} the highest ordering was in {} to reach {} $ </b>".format(' and '.join(map(str,Country_filter)),pd.to_datetime(str(max_year_order)).strftime('%Y.%m.%d'),round(maxvalue_order,2)),unsafe_allow_html=True)
        col2.markdown("<b>In {} the lowest ordering was in {} to reach {} $ </b>".format(' and '.join(map(str,Country_filter)),pd.to_datetime(str(min_year_order)).strftime('%Y.%m.%d'),round(minvalue_order,2)),unsafe_allow_html=True)

        col4.markdown("<b>In {} the highest Shipping cost was in {} to reach {} $ </b>".format(' and '.join(map(str,Country_filter)),pd.to_datetime(str(max_year_cost)).strftime('%Y.%m.%d'),round(maxvalue_cost,2)),unsafe_allow_html=True)
        col4.markdown("<b>In {} the lowest Shipping cost was in {} to reach {} $ </b> \n <br>".format(' and '.join(map(str,Country_filter)),pd.to_datetime(str(min_year_cost)).strftime('%Y.%m.%d'),round(minvalue_cost,2)),unsafe_allow_html=True)
        st.markdown("<h2> Variations of Orders with respect to Priority ,Segments ,and Product Category",unsafe_allow_html=True)
        col1,col2,col3,col4=st.beta_columns((1,2,2,1))
        a1=data.loc[(data['Country'].isin(Country_filter))]
        Priority=a1.groupby('Order Priority')['Order ID'].count().reset_index()

        labels=Priority['Order Priority']
        values=Priority['Order ID']
        
        fig1=go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6,marker_colors=[	'red','#A9A9A9'])])
        fig1.update_layout(title="% of Order Priority Distribution".format(Country_filter),annotations=[dict(text='{}'.format(' <br>'.join(map(str,Country_filter))),x=0.48,y=0.5,font_size=25,showarrow=False)],width=400,height=400)

        # fig1=px.pie(Priority,values='Order ID',names='Order Priority')
        # fig1.update_layout(title='% of Alcohol use among youth in {}'.format(labels),annotations=[dict(text='{}'.format(Country_filter),x=0.48,y=0.5,font_size=20,showarrow=False)],width=400,height=400)

        col2.plotly_chart(fig1)
        Segment=a1.groupby('Segment')['Order ID'].count().reset_index()
        labels=Segment['Segment']
        values=Segment['Order ID']
        
        fig2=go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6,marker_colors=[	'red','#A9A9A9'])])
        fig2.update_layout(title=" % of Order Segment Distribution".format(Country_filter),annotations=[dict(text='{}'.format(' <br>'.join(map(str,Country_filter))),x=0.48,y=0.5,font_size=25,showarrow=False)],width=400,height=400)
        st.markdown("<br> \n",unsafe_allow_html=True)
        col3.plotly_chart(fig2)
        
        Category=a1.groupby('Category')['Order ID'].count().reset_index()

        labels3=Category['Category']
        values3=Category['Order ID']
        
        fig3=go.Figure(data=[go.Pie(labels=labels3, values=values3, hole=0.6,marker_colors=[	'red','#A9A9A9'])])
        fig3.update_layout(title="% of Order Category Distribution".format(Country_filter),annotations=[dict(text='{}'.format(' <br>'.join(map(str,Country_filter))),x=0.48,y=0.5,font_size=25,showarrow=False)],width=400,height=400)

        # fig1=px.pie(Priority,values='Order ID',names='Order Priority')
        # fig1.update_layout(title='% of Alcohol use among youth in {}'.format(labels),annotations=[dict(text='{}'.format(Country_filter),x=0.48,y=0.5,font_size=20,showarrow=False)],width=400,height=400)

        col4.plotly_chart(fig3)
        
    
if navigation=='Predictive Analysis':
    st.title("Predicting profit using Machine Learning Regression Models")

    st.markdown("<h2> Let's discover the variables correlation in this dataset <br>",unsafe_allow_html=True)
    X=data[['Ship Mode','Segment','City','Region','Order Priority','Quantity','Discount']]
    Y=data['Profit']
    X['Ship Mode'] = X['Ship Mode'].astype('category')
    X['Segment'] = X['Segment'].astype('category')
    X['City'] = X['City'].astype('category')
    X['Region'] = X['Region'].astype('category')
    X['Order Priority'] = X['Order Priority'].astype('category')

    X['Ship Mode']  = X['Ship Mode'] .cat.codes
    X['Segment']  = X['Segment'] .cat.codes
    X['City']  = X['City'] .cat.codes
    X['Region']  = X['Region'] .cat.codes
    X['Order Priority']  = X['Order Priority'] .cat.codes
    col1,col2,col3=st.beta_columns((1,2,1))
    corr = data.corr(method='pearson')
    plt.close()
    fig, ax = plt.subplots(figsize=(9,10))
    cor_plot = sns.heatmap(corr,annot=True,linewidths=.8,cmap='RdYlGn',ax=ax)
    fig=plt.gcf()
    plt.xticks(fontsize=15,rotation=-30)
    plt.yticks(fontsize=15)
    plt.title('Correlation Matrix')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col2.pyplot(fig)

    st.markdown("<h2> Data Preprocessing <br>",unsafe_allow_html=True)
    st.markdown("""1. We chose the following columns to be our predictors in the Machine Learning Models: <br> X=data[['Ship Mode','Segment','City','Region','Order Priority','Quantity','Discount']]
<br> Y=data['Profit']""",unsafe_allow_html=True)
    st.markdown("""2. Most of the predictors are Categorical so we did some standard Scalar transformation in order to normalize our data so we would get a better result in the machine learning models 
    <br> scaler_train = StandardScaler() <br>
scaler_train.fit(x_train)""",unsafe_allow_html=True)
    st.markdown("""3. Finaly before starting training the models, we splitted the data into training and testing <br>
    the training data is 80% while teh testing data is 20% <br> from sklearn.model_selection import train_test_split <br>
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size = .2,
                                                    random_state = 1)""",unsafe_allow_html=True)
    X=data[['Ship Mode','Segment','City','Region','Order Priority','Quantity','Discount']]
    Y=data['Profit']
    X['Ship Mode'] = X['Ship Mode'].astype('category')
    X['Segment'] = X['Segment'].astype('category')
    X['City'] = X['City'].astype('category')
    X['Region'] = X['Region'].astype('category')
    X['Order Priority'] = X['Order Priority'].astype('category')

    X['Ship Mode']  = X['Ship Mode'] .cat.codes
    X['Segment']  = X['Segment'] .cat.codes
    X['City']  = X['City'] .cat.codes
    X['Region']  = X['Region'] .cat.codes
    X['Order Priority']  = X['Order Priority'] .cat.codes
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size = .2,
                                                        random_state = 1)
    from sklearn import preprocessing, svm
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    scaler_train = StandardScaler()
    scaler_train.fit(x_train)
    X_main_scaled = scaler_train.transform(x_train)
    X_main_scaled_test = scaler_train.transform(x_test)
    # Splitting the data into training and testing data
    linear_regr = LinearRegression()
    linear_regr.fit(X_main_scaled, y_train)
    y_pred = linear_regr.predict(X_main_scaled_test)
    accuracy = linear_regr.score(X_main_scaled,y_train)




    st.markdown("<h2> Machine Learning Models <br>",unsafe_allow_html=True)
    st.markdown("<h3> Linear Regression: <br>",unsafe_allow_html=True)
    st.write("Train Accuracy {}%".format(int(round(accuracy *100))))
    st.write("Testing RMSE Linear regression ",mean_squared_error(y_test, y_pred))
    st.markdown("<h3> Decision Tree: <br>",unsafe_allow_html=True)
    from sklearn.tree import DecisionTreeRegressor 
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    # create a regressor object
    dt_regr = DecisionTreeRegressor(random_state = 0) 
    
    # fit the regressor with X and Y data
    dt_regr.fit(X_main_scaled, y_train)
    y_pred = dt_regr.predict(X_main_scaled_test)
    accuracy = dt_regr.score(X_main_scaled,y_train)
    st.write("Train Accuracy {}%".format(int(round(accuracy *100))))
    st.write("Testing RMSE Decision Tree regressor ",mean_squared_error(y_test, y_pred))
    st.markdown("<h3> Random Forest: <br>",unsafe_allow_html=True)
    clf = RandomForestRegressor(
                                random_state=1)
    clf.fit(X_main_scaled, y_train)
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    # st.write("Train Accuracy {}%".format(int(round(accuracy *100))))
    # st.write("Training RMSE Linear regression ",mean_squared_error(y_test, y_pred))
      
    y_pred = clf.predict(X_main_scaled_test)
    accuracy = clf.score(X_main_scaled,y_train)
    st.write("Train Accuracy {}%".format(int(round(accuracy *100))))
    st.write("Testing RMSE Decision Tree regressor ",mean_squared_error(y_test, y_pred))
   
    st.markdown("<h2> The models used showed high RMSE on the testing set which means that the models failed to predict the profits <br> We recomend adding more variables to achieve a better accuracy ",unsafe_allow_html=True)

