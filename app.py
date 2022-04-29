# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:24:10 2022

@author: Shubham Shah
"""
import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from pyvis.network import Network
import streamlit.components.v1 as components
import pickle
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@st.cache
def load_data():
    df = pd.read_csv('./data/final_2020.csv')
    origin = pd.read_csv('./LookupFiles/Origin_lookup.csv')
    origin = origin.set_index('Code')
    dest = pd.read_csv('./LookupFiles/Dest_lookup.csv')
    dest = dest.set_index('Code')
    return df, origin, dest


# load the data
df, origin, dest = load_data()
originList, destList = list(df['ORIGIN'].unique()), list(df['DEST'].unique())
originList.insert(1,'AL')
destList.insert(1,"AL")
page_list = ["Network Graph","Comparison", "Relation", "Distribution Graph", "Feature Importance", "Price Prediction"]
categorical = ['QUARTER','AIRLINE_COMPANY', 'ORIGIN', 'ORIGIN_WAC', 'DEST', 'DEST_WAC', 'AIRPORT_GROUP', 'WAC_GROUP', 'DISTANCE_GROUP', 'CONTINUE_USA']
numerical = ['PRICE_PER_TICKET','MARKET_COUPONS', 'MILES', 'NUM_OF_TICKETS']
airlines = list(df['AIRLINE_COMPANY'].unique())
features = ['MARKET_COUPONS', 'QUARTER', 'ORIGIN', 'DEST', 'MILES', 'NUM_OF_TICKETS', 'AIRLINE_COMPANY']
airlines.insert(0,'ALL')
miles = df[['ORIGIN', 'DEST','MILES']]
miles = miles.drop_duplicates()
# choose the sources of interest
cols = list(df.columns)
origin_dict = pd.read_csv("./LookupFiles/origin_dict.csv").set_index('Unnamed: 0')
dest_dict = pd.read_csv("./LookupFiles/dest_dict.csv").set_index('Unnamed: 0')
airline_dict = pd.read_csv("./LookupFiles/airline_dict.csv").set_index('Unnamed: 0')
airline_vals = ['WN','AA','DL','UA','AS','B6','NK','G4']
def feature_helper(importances):
    fig, ax = plt.subplots(figsize=(10,6), facecolor = '#f0faf3')
    ax.set(facecolor='#b4b4b4')
    n=12
    imp = importances.sort_values()[-n:].reset_index()
    st.altair_chart(alt.Chart(imp).mark_bar().encode(
    x='0:Q',
    y='index:N'
))
    
def multiselect_helper(airline, df):
        if 'ALL' not in airline:
            df1 = df[df['AIRLINE_COMPANY'].isin(airline)]
        else:
            df1=df
        source=st.selectbox('Source', originList,1, format_func=lambda x: origin.loc[x]['Description'])
        destination=st.selectbox('Destination', destList, 1, format_func=lambda x: dest.loc[x]['Description'])
        if source!='AL' and destination!='AL':
            df2 = df1[(df1['ORIGIN']==source) & (df1['DEST']==destination)]
        elif source!='AL' and destination=='AL':
            df2 = df1[(df1['ORIGIN']==source)]
        elif source=='AL' and destination!='AL':
            df2 = df1[df1['DEST']==destination]
        else:
            df2 = df1
        return df2

def comparison():
    if st.sidebar.checkbox('Show dataframe', value=True):
        option = st.multiselect('What sources do you want to display?', cols, cols)
        st.write("As dataset is huge, we have limited the rows to 10000.")
        st.write(df[option].head(10000))
    if st.sidebar.checkbox('Show Bar Charts'):
        st.title("Comparison using Bar Charts")
        field1=st.selectbox('Categorical Variable: Field 1', categorical)
        field2=st.selectbox('Numerical Variable: Field 2', numerical)
        
        radio = st.radio("Do you want to select multiple source-destination pairs?",('No', 'Yes'))
        st.write("Select ALL in both source and destination to get barchart of whole dataset. \nNote: It will take time as dataset is very large.")
        if radio=='No':
            source=st.selectbox('Source', originList, format_func=lambda x: origin.loc[x]['Description'])
            destination=st.selectbox('Destination', destList, format_func=lambda x: dest.loc[x]['Description'])
            source = [source]
            destination = [destination]
        else:
            source=st.multiselect('Source', originList, format_func=lambda x: origin.loc[x]['Description'])
            destination=st.multiselect('Destination', destList, format_func=lambda x: dest.loc[x]['Description'])
            if len(source) != len(destination):
                st.write("Please select same number of source and destination and length of both should be greater than zero.")
                return
            elif len(source)==0:
                st.write("Select the source.")
            elif len(destination) == 0:
                st.write("Select the destination.")
            elif source == destination:
                st.write("Source and Destination same.")
        srcDest = list(zip(source, destination))
        if ('AL', 'AL') not in srcDest:    
            df1 = df[df[['ORIGIN', 'DEST']].apply(tuple, axis=1).isin(srcDest)]
        else:
            df1 = df.head(10000)
            
        # Replace df1.head(100000) if len(df1)>100000 else df1 with df1 to consider whole dataset.
        st.altair_chart(alt.Chart(df1.head(100000) if len(df1)>100000 else df1).mark_bar().encode(
                        x=field1+':O',
                        y='mean('+field2+'):Q',
                        tooltip=[field1, 'mean('+field2+')']
                        ).interactive(), use_container_width=True)
        st.write("Note: The total rows are limited to first 10000 as it will take a lot of time to generate chart")
    else:
        st.write("Select checkbox on left panel to see the results.")
        
def relation():
    if st.sidebar.checkbox('Show dataframe'):
        option = st.multiselect('What sources do you want to display?', cols, cols)
        st.write("As dataset is huge, we have limited the rows to 10000.")
        st.write(df[option].head(10000))
        
    if st.sidebar.checkbox('Show Line chart and Scatter Plots'):
        st.title("Relation using Line Charts")
        field1=st.selectbox('Numerical Variable: Field 1', numerical)
        field2=st.selectbox('Numerical Variable: Field 2', numerical, len(numerical)-1)
        if field1 == field2:
            st.write("Both fields are selected same please try two different fields.")
            return 
        airline = st.multiselect('Airline', airlines, ['ALL'])
        df2 = multiselect_helper(airline, df)
        # Replace df1.head(100000) if len(df1)>100000 else df1 with df1 to consider whole dataset.
        st.altair_chart(alt.Chart(df2.head(100000) if len(df2)>100000 else df2).mark_line().encode(
                        x=field1,
                        y=field2,
                        tooltip=[field1, field2],
                        color='AIRLINE_COMPANY',
                        strokeDash='AIRLINE_COMPANY'
                        ).interactive(), use_container_width=True)
        st.title("Scatter Plot")
        st.altair_chart(alt.Chart(df2.head(100000) if len(df2)>100000 else df2).mark_circle(size=60).encode(
                        x=field1,
                        y=field2,
                        tooltip=[field1, field2],
                        color='AIRLINE_COMPANY',
                        strokeDash='AIRLINE_COMPANY'
                        ).interactive(), use_container_width=True)
        st.write("Note: The total rows are limited to first 100000 as it will take a lot of time to generate chart")
        
       
def distribution_graph():
    field=st.selectbox('Numerical Variable: Field', numerical)
    st.altair_chart(alt.Chart(df.head(10000)).transform_density(field, as_=[field, 'DENSITY'],
                             ).mark_area(color='green').encode(
    x=field+":Q",
    y='DENSITY:Q',
    ).interactive(), use_container_width=True)
    st.write("Note: The total rows are limited to first 10000 as it will take a lot of time to generate chart")

def feature_importance():
    st.title("Which feature have the most influence on the price?")
    airline = st.multiselect('Airline', airlines, ['ALL'])
    df2 = multiselect_helper(airline, df)
    if len(df2) == len(df):
        model = pickle.load(open('./model/finalized_model.sav', 'rb'))
        importances = pd.Series(model.feature_importances_, list(features))
        feature_helper(importances)
    else:
        X = df[features]
        Y = df['PRICE_PER_TICKET']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=99)
        pipeline = make_pipeline(
                OrdinalEncoder(),
                RandomForestRegressor(max_depth=20, n_estimators=20)
            )
            
        pipeline.fit(X_train, Y_train)
        model = pipeline.named_steps['randomforestregressor']
        importances = pd.Series(model.feature_importances_, list(X.columns))
        feature_helper(importances)
        
def prediction():
    source=st.selectbox('Source', originList, format_func=lambda x: origin.loc[x]['Description'])
    destination=st.selectbox('Destination', destList, format_func=lambda x: dest.loc[x]['Description'])
    airline=st.selectbox('AirLine', airline_vals)
    distance = miles[(miles['ORIGIN']==source) & (miles['DEST']==destination)]
    if not distance.empty:
        st.write("Total Miles:", distance['MILES'].to_list()[0])
    else:
        st.write("This combination is not available.")
        return
    d = st.date_input(
         "When do you want to travel?",
         datetime.datetime.now())
    quarter = (d.month-1)//4 + 1
    market_coupons = int(st.slider('How many Market Coupons?', 0, 100, 0))
    num_tickets = int(st.slider('Number of Tickets?', 1, 100, 1))
    source = origin_dict.loc[source, '0']
    destination =  dest_dict.loc[destination, '0']
    airline = airline_dict.loc[airline,'0']
    model = pickle.load(open('./model/finalized_model.sav', 'rb'))
    test = pd.DataFrame({ 'ORIGIN':[source], "DEST":[destination], "AIRLINE_COMPANY":[airline], "QUARTER":[quarter], 'MILES':float(distance['MILES'].to_list()[0]), 'MARKET_COUPONS': market_coupons, 'NUM_OF_TICKETS': num_tickets 
        })
    predicted = model.predict(test)[0]
    st.write("The predicted price is :",predicted)
    importances = pd.DataFrame({'Percentage Increase':model.feature_importances_, 'Because of':list(features)})
    importances['Percentage Increase'] = importances["Percentage Increase"].apply(lambda x: x * 100)
    importances['Price Increased'] = importances['Percentage Increase'].apply(lambda x: x * predicted/100)
    st.altair_chart(alt.Chart(importances).mark_arc().encode(
        theta=alt.Theta(field="Percentage Increase", type="quantitative"),
        color=alt.Color(field="Because of", type="nominal"),
        tooltip=['Because of', 'Percentage Increase', 'Price Increased']
        ))

    
    
    
def network_graph():
    radio_options = ("Average between Same Source and Destination","Average Per Airlines",'First 100 rows', 'First 1000 rows', 'First 10000 rows', 'All')
    radio = st.sidebar.radio("Need to limit number of rows as dataset is very big",radio_options)
    rows=None
    if radio in radio_options[2:]:    
        rows = 100 if radio=="First 100 rows" else 1000 if radio=="First 1000 rows" else 10000 if radio=="First 10000 rows" else None
        df1 = df
    elif radio == radio_options[0]:
        df1 = df[['ORIGIN','DEST','MILES', 'PRICE_PER_TICKET']].groupby(['ORIGIN', 'DEST']).mean().reset_index()
    elif radio == radio_options[1]:
        df1 = df[['ORIGIN','DEST','MILES', 'PRICE_PER_TICKET', 'AIRLINE_COMPANY']].groupby(['ORIGIN', 'DEST', 'AIRLINE_COMPANY']).mean().reset_index()
    st.title("Air travel Network Graphs")
    airline = st.multiselect('Airline', airlines, ['ALL'])
    df2 = multiselect_helper(airline, df1)
    df3 = df2
    
    got_net = Network(directed=True, width='100%')
    got_net.barnes_hut()
    origin = pd.read_csv('./LookupFiles/Origin_lookup.csv')
    origin = origin.set_index('Code')
    dest = pd.read_csv('./LookupFiles/Dest_lookup.csv')
    dest = dest.set_index('Code')
    # got_data = got_data[got_data['AIRLINE_COMPANY']==airline] if airline else got_data
    # got_data = got_data.sample(n=10000) if len(got_data) > 10000 else got_data
    if rows:
        df3 = df2.head(rows)
    print(df2.head())
    nums = 100
    sources = df3.head(nums)['ORIGIN']
    targets = df3.head(nums)['DEST']
    miles = df3.head(nums)['MILES']
    prices = df3.head(nums)['PRICE_PER_TICKET']
    if rows:
        als = df3.head(nums)['AIRLINE_COMPANY']
        edge_data = zip(sources, targets, miles, prices,als)
    else:
        edge_data = zip(sources, targets, miles, prices)
    for e in edge_data:
        src = origin.loc[e[0]]['Description']
        dst = dest.loc[e[1]]['Description']
        m = e[2]
        p = e[3]
        if rows:
            a = e[4]
        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
        t = "#Miles:"+str(m)+" | #Price:"+str(p)+(" | #AirLine:"+a if rows else '')
        got_net.add_edge(src, dst, title=t)
    
    neighbor_map = got_net.get_adj_list()
    
    for node in got_net.nodes:
        node['title'] += ' Flights To:<br>' + '<br>'.join(neighbor_map[node['id']])
        node['value'] = len(neighbor_map[node['id']])
    
    
    # got_net.set_edge_smooth('dynamic')
    got_net.set_options("""
                        var options = {
      "nodes": {
        "color": {
          "highlight": {
            "border": "rgba(0,0,0,1)",
            "background": "rgba(0,0,0,1)"
          }
        }
      },
      "edges": {
        "color": {
          "highlight": "rgba(0,0,0,1)",
          "inherit": false
        },
        "smooth": {
          "forceDirection": "none"
        }
      },
      "interaction": {
        "multiselect": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "springLength": 250,
          "springConstant": 0.001
        },
        "minVelocity": 0.75
      }
    }""")
    # got_net.show_buttons()
    # got_net.show('networkGraph.html')
    
    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
       path = './tmp'
       got_net.save_graph(f'{path}/networkGraph.html')
       HtmlFile = open(f'{path}/networkGraph.html','r',encoding='utf-8')
    # Save and read graph as HTML file (locally)
    except:
        path = './html_files'
        got_net.save_graph(f'{path}/networkGraph.html')
        HtmlFile = open(f'{path}/networkGraph.html','r',encoding='utf-8')
    # Load HTML into HTML component for display on Streamlit
    components.html(HtmlFile.read(), height = 1200, width=1000)
    st.write("Note: The total rows are limited to first 100 as it will take a lot of time to generate chart")

st.sidebar.title("Air travel Price Prediction")
selected=st.sidebar.selectbox('App Navigation', page_list)

if selected=="Comparison":
    comparison()
elif selected=="Relation":
    relation()
elif selected=="Network Graph":
    network_graph()
elif selected=="Distribution Graph":
    distribution_graph()
elif selected=='Feature Importance':
    feature_importance()
elif selected=='Price Prediction':
    prediction()