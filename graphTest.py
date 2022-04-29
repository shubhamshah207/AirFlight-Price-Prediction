from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd

# airline = None
got_net = Network(width="100%", height="100%", directed=True)
got_net.barnes_hut()
got_data = pd.read_csv('./data/final_2020.csv')
origin = pd.read_csv('./LookupFiles/Origin_lookup.csv')
origin = origin.set_index('Code')
dest = pd.read_csv('./LookupFiles/Dest_lookup.csv')
dest = dest.set_index('Code')
# got_data = got_data[got_data['AIRLINE_COMPANY']==airline] if airline else got_data
# got_data = got_data.sample(n=10000) if len(got_data) > 10000 else got_data
got_data = got_data.head(3) if len(got_data) > 3 else got_data
sources = got_data['ORIGIN']
targets = got_data['DEST']
miles = got_data['MILES']
prices = got_data['PRICE_PER_TICKET']
airlines = got_data['AIRLINE_COMPANY']
edge_data = zip(sources, targets, miles, prices,airlines)

for e in edge_data:
    src = origin.loc[e[0]]['Description']
    dst = dest.loc[e[1]]['Description']
    m = e[2]
    p = e[3]
    a = e[4]
    got_net.add_node(src, src, title=src)
    got_net.add_node(dst, dst, title=dst)
    t = "#From:"+src+" | #To:"+dst+" | #Miles:"+str(m)+" | #Price:"+str(p)+" | #AirLine:"+a
    got_net.add_edge(src, dst, title=t)

neighbor_map = got_net.get_adj_list()

for node in got_net.nodes:
    node['title'] += ' Flights To:<br>' + '<br>'.join(neighbor_map[node['id']])
    node['value'] = len(neighbor_map[node['id']])


# got_net.set_edge_smooth('dynamic')
# got_net.set_options("""
#                     var options = {
#   "nodes": {
#     "color": {
#       "highlight": {
#         "border": "rgba(0,0,0,1)",
#         "background": "rgba(0,0,0,1)"
#       }
#     }
#   },
#   "edges": {
#     "color": {
#       "highlight": "rgba(0,0,0,1)",
#       "inherit": false
#     },
#     "smooth": {
#       "forceDirection": "none"
#     }
#   },
#   "interaction": {
#     "multiselect": true
#   },
#   "physics": {
#     "barnesHut": {
#       "gravitationalConstant": -80000,
#       "springLength": 250,
#       "springConstant": 0.001
#     },
#     "minVelocity": 0.75
#   }
# }""")
got_net.show_buttons()
got_net.show('networkGraph.html')

# # Save and read graph as HTML file (on Streamlit Sharing)
# try:
#    path = './tmp'
#    got_net.save_graph(f'{path}/networkGraph.html')
#    HtmlFile = open(f'{path}/networkGraph.html','r',encoding='utf-8')
# # Save and read graph as HTML file (locally)
# except:
#     path = './html_files'
#     got_net.save_graph(f'{path}/networkGraph.html')
#     HtmlFile = open(f'{path}/networkGraph.html','r',encoding='utf-8')

