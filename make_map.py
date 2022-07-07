from color import colors, number

import folium 
import warnings 
import osmnx as ox
import os 

warnings.filterwarnings('ignore')



def generate_main_html(main_road_nodes, main_road_edges, place):
    region =  place.split(' ')[0]
    
    places = ox.geocode_to_gdf([place])
    places = ox.project_gdf(places)

    #lat, lon
    latitude = places.lat.values[0]; longitude = places.lon.values[0]
    #기본 지도 정의
    m = folium.Map(location=[latitude, longitude],
                zoom_start=11)

    #법정경계 표시
    folium.Choropleth(geo_data=places.geometry,
                    fill_color="white",
                    ).add_to(m)

    #Nodes
    for i in range(len(main_road_nodes)):
        folium.CircleMarker([main_road_nodes.iloc[[i]]["geometry"].values[0].y,main_road_nodes.iloc[[i]]["geometry"].values[0].x],
                            color = colors[number(main_road_nodes.iloc[[i]].cluster.values[0])],
                            radius = 3
                        ).add_to(m)

    for i in range(len(main_road_edges)):
        folium.Choropleth(
            main_road_edges.iloc[[i]]["geometry"],
            line_weight = 3,
            line_color = colors[number(main_road_edges.iloc[[i]].cluster.values[0])] if type(main_road_edges.iloc[[i]].cluster.values[0]) != list else "black"
        ).add_to(m)
        
    os.chdir('/home/yh_zoo/Graph_segment/result/')
    os.mkdir(f'./html/{region}')
    m.save(f'./html/{region}/main.html')
    
    
def generate_non_main_html(non_main_road_edges, place):
    region =  place.split(' ')[0]
    
    places = ox.geocode_to_gdf([place])
    places = ox.project_gdf(places)

    #lat, lon
    latitude = places.lat.values[0]; longitude = places.lon.values[0]
    #기본 지도 정의
    m = folium.Map(location=[latitude, longitude],
                zoom_start=11)

    #법정경계 표시
    folium.Choropleth(geo_data=places.geometry,
                    fill_color="white",
                    ).add_to(m)

    for i in range(len(non_main_road_edges)):
        folium.Choropleth(
            non_main_road_edges.iloc[[i]]["geometry"],
            line_weight = 1 if non_main_road_edges.iloc[[i]].M_category.values[0] == 9999 else 4,
            line_color = colors[number(non_main_road_edges.iloc[[i]].M_category.values[0])]
        ).add_to(m)
    
    os.chdir('/home/yh_zoo/Graph_segment/result/')
    m.save(f'./html/{region}/non_main.html')    