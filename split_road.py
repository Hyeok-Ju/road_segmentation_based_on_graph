import os
import geopandas as gpd
import pandas as pd 
import numpy as np
import folium 
import itertools 
import pickle
from tqdm import tqdm 
import warnings 

warnings.filterwarnings('ignore')

from line_extension import linestring_extension, tunnel_linestring_extension


###간선도로(main_road), 집산도로(non_main_road) 분류
def split_based_on_mainroad(data_nodes, data_edges):   
    data_edges["idx"] = range(len(data_edges))
    ###edges filter
    #main_road_edges filter
    primary = data_edges.loc[list(map(lambda data: "primary" in data, data_edges.highway))] #주간선도로
    secondary =  data_edges.loc[list(map(lambda data: "secondary" in data, data_edges.highway))] #간선도로
    tertiary = data_edges.loc[list(map(lambda data: "tertiary" in data, data_edges.highway))] #보조간선도로
    main_road = pd.concat([primary, secondary, tertiary])
    main_road_edges_mask = set(main_road.idx)
    main_road_edges = data_edges.iloc[sorted(main_road_edges_mask)]
    
    #non_main_road_edges filter
    non_main_road_mask = set(data_edges.idx) - set(main_road_edges.idx)
    non_main_road_edges = data_edges.iloc[sorted(non_main_road_mask)]
    
    connect_node_inf = set(main_road_edges[['u','v']].values.reshape(-1)) & set(non_main_road_edges[['u','v']].values.reshape(-1))
        
    ###nodes filter
    main_road_nodes_mask = set(main_road_edges[['u','v']].values.reshape(-1))
    non_main_road_nodes_mask = set(non_main_road_edges[['u','v']].values.reshape(-1))

    main_road_nodes = data_nodes.loc[[i in main_road_nodes_mask for i in data_nodes['osmid']]]
    non_main_road_nodes = data_nodes.loc[[i in non_main_road_nodes_mask for i in data_nodes['osmid']]]

    main_road_edges = main_road_edges.drop(["idx"],axis=1)
    non_main_road_edges = non_main_road_edges.drop(["idx"],axis=1)
    non_main_road_nodes["connect_inf"] = [1 if i else 0 for i in list(map(lambda data : data in connect_node_inf, non_main_road_nodes['osmid']))]
    return [main_road_nodes, main_road_edges, connect_node_inf], [non_main_road_nodes, non_main_road_edges]


###노드 추가를 위한 노드 ID 재정의
def generate_new_node_id(data_nodes, data_edges): 
    node_osmid_dict = dict()
    # if "connect_inf" in data_nodes.columns:
    #     data_nodes = data_nodes.sort_values(by='connect_inf', ascending=False)
    for idx, i in enumerate(list(data_nodes['osmid'])):
        node_osmid_dict[i] = idx
    ###data_nodes -> node_id 재정의 
    data_nodes['osmid'] = [node_osmid_dict[i] for i in data_nodes['osmid']]

    ###data_edges -> node_id 재정의 
    data_edges['u'] = [node_osmid_dict[i] for i in data_edges['u']]
    data_edges['v'] = [node_osmid_dict[i] for i in data_edges['v']]
    return data_nodes, data_edges, node_osmid_dict

# 터널 노드 정보 합치기 및 노드 ID 초기화
# input : [data_1 : road] , [data_2 : tunnel]
def combine_data(data_1, data_2):
    base_nodes, base_edges = data_1
    tunnel_nodes, tunnel_edges = data_2
    
    #road에 속해있지 않은 터널은 제거
    mask = set(tunnel_edges[["geometry"]].sjoin(base_edges[["geometry"]],how="left",op="within").dropna().index)
    tunnel_edges = tunnel_edges.loc[mask]
    tunnel_nodes = tunnel_nodes.loc[[i in set(tunnel_edges[['u','v']].values.reshape(-1)) for i in tunnel_nodes['osmid']]]

    mask = set(base_nodes['osmid']) & set(tunnel_nodes['osmid'])
    base_nodes = base_nodes.loc[[i in set(set(base_nodes['osmid'])- set(tunnel_nodes['osmid'])) for i in base_nodes['osmid']]]
    
    # state = Cross : 1, tunnel : 2
    base_nodes["state"] = 1
    tunnel_nodes['state'] = 2

    base_nodes = pd.concat([base_nodes, tunnel_nodes])

    nodes, edges, id_dict = generate_new_node_id(base_nodes, base_edges)
    mask = set([id_dict[i] for i in mask])
    return [nodes, edges, mask]


###끝 점 state 변경 함수
# input : [road_nodes, road_edges]
def end_point_check(data_nodes, data_edges):
    nodes_count = pd.DataFrame(pd.DataFrame(data_edges[['u','v']].values.reshape(-1)).value_counts(), columns=['cnt']).reset_index()
    nodes_count.columns = ['osmid', 'cnt']

    end_point = set(nodes_count.loc[(nodes_count['cnt'] == 1).values]['osmid'].values)
    end_point_mask = []
    
    for idx, state in zip(data_nodes['osmid'], data_nodes['state']):
        if (idx in end_point) and (state == 1):
            end_point_mask.append(-1)
        else:
            end_point_mask.append(state)

    data_nodes.state = end_point_mask
    return data_nodes



def feature_sort(main, non_main):
    '''
    도시마다 컬럼 순서가 달라서 고정! 
    '''
    main_nodes, main_edges = main
    non_main_nodes, non_main_edges = non_main
    
    main_nodes_features = ['osmid', 'y', 'x', 'street_count', 'highway', 'geometry', 'state']
    non_main_nodes_features = ['osmid', 'y', 'x', 'street_count', 'highway', 'geometry', 'state', 'connect_inf']
    edges_features = ['u', 'v', 'key', 'osmid', 'oneway', 'lanes', 'highway', 'maxspeed', 'length', 'geometry',
                    'name', 'bridge', 'tunnel', 'service', 'access', 'junction','width']
    
    main_nodes = main_nodes[main_nodes_features]
    main_edges = main_edges[edges_features]
    non_main_nodes = non_main_nodes[non_main_nodes_features]
    non_main_edges = non_main_edges[edges_features]
    
    return main_nodes, main_edges, non_main_nodes, non_main_edges


def split_road_and_change_nodes_state(road, tunnel):
    '''
    1 : 간선도로, 집산도로 분리  
    2 : 분리 후 없어져도 되는 node 제거 후 linestring 연장
    '''
    road_nodes, road_edges = road 
    tunnel_nodes, tunnel_edges = tunnel
    
    main_road, non_main_road = split_based_on_mainroad(road_nodes, road_edges)          # 1
    main_road_nodes, main_road_edges = linestring_extension(main_road)                  # 2

    main_tunnel, non_main_tunnel = split_based_on_mainroad(tunnel_nodes, tunnel_edges)  # 1  
    main_tunnel_nodes, main_tunnel_edges = tunnel_linestring_extension(main_tunnel[:2]) # 2


    '''
    자세한 nodes 터널 정보 추가
    '''
    # 간선도로
    main_road_nodes, main_road_edges, main_road_mask= combine_data([main_road_nodes, main_road_edges], [main_tunnel_nodes, main_tunnel_edges])
    # 집산도로 
    non_main_road_nodes, non_main_road_edges, non_main_road_mask = combine_data(non_main_road, non_main_tunnel)

    '''
    nodes state(상태) 세부분류 -> 끝점 -1, 교차로 1, 터널 2
    '''
    main_road_nodes = end_point_check(main_road_nodes, main_road_edges)
    non_main_road_nodes = end_point_check(non_main_road_nodes, non_main_road_edges)


    '''
    도시 별로 features의 순서가 달라서 고정
    '''
    main_road_nodes, main_road_edges, non_main_road_nodes, non_main_road_edges = \
        feature_sort([main_road_nodes, main_road_edges],[non_main_road_nodes, non_main_road_edges]) 


    return [main_road_nodes, main_road_edges, main_road_mask], [non_main_road_nodes, non_main_road_edges, non_main_road_mask]