### 0. Packages to use 
import osmnx as ox 
import os
import copy
import pandas as pd 
import numpy as np
import itertools 
import pickle
import warnings 

warnings.filterwarnings('ignore')



def graph_from_place(place):
    '''
    * OSM graph load (tunnel의 자세한 위치를 얻기 위해 simplify=T/F의 두 개의 데이터를 모두 load)
        - 같은 지역을 불러올때 시간 단축을 위해 local에 저장하면서함
    '''
    region =  place.split(' ')[0]

    os.chdir('/home/yh_zoo/Graph_segment/rawdata/')
    rawdata_list = os.listdir()

    if region in rawdata_list:     
        # base nodes, edges
        G1_nodes = pd.read_pickle(f'./{region}/nodes_ST.pkl')
        G1_edges = pd.read_pickle(f'./{region}/edges_ST.pkl')
        # nodes, edges for tunnel
        G2_nodes = pd.read_pickle(f'./{region}/nodes_SF.pkl')
        G2_edges = pd.read_pickle(f'./{region}/edges_SF.pkl')
    else: 
        G1 = ox.graph_from_place(place, network_type="drive_service", simplify=True)
        G2 = ox.graph_from_place(place, network_type="drive_service", simplify=False)
        # base nodes, edges
        os.mkdir(f'{region}')
        G1_nodes, G1_edges = ox.graph_to_gdfs(G1)
        G1_nodes.to_pickle(f'./{region}/nodes_ST.pkl')
        G1_edges.to_pickle(f'./{region}/edges_ST.pkl')
        # nodes, edges for tunnel
        G2_nodes, G2_edges = ox.graph_to_gdfs(G2)
        G2_nodes.to_pickle(f'./{region}/nodes_SF.pkl')
        G2_edges.to_pickle(f'./{region}/edges_SF.pkl')
    return [G1_nodes, G1_edges], [G2_nodes, G2_edges]

#########################################################################################
def remove_overlap_roads(data_edges):
    '''
    *겹치는 도로 제거 
        - OSM에서는 양방향성을 고려하여 겹쳐있는 도로가 있지만, 
          위험도 분석에서는 겹치는 양방향성 도로가 필요 없기때문에 제거함
    '''
    
    edges = copy.deepcopy(data_edges)
    edges["idx"] = range(len(edges))
        
    def list_check(data):
        return 1 if type(data) == list else 0    
        
    def geometry_convert_list(data):
        return list(itertools.chain(*data.coords[:]))

    def convert_list_to_tuple(data):
        return tuple(sorted(data)) if type(data) == list else data 
        
    edges["geometry"] = list(map(lambda data: geometry_convert_list(data), edges["geometry"].values.tolist()))
    
    for i in edges.columns:
        check = sum(list(map(lambda data: list_check(data), edges[i].values.tolist())))
        if check > 0:
            edges[i] = list(map(lambda data: convert_list_to_tuple(data), edges[i].values.tolist()))
                
    edges = edges.drop_duplicates(subset=['geometry'])

    print(f"- remove_undirected : rink {len(data_edges)-len(edges)}개 삭제")   
    return data_edges.iloc[edges.idx.values.tolist()]



#########################################################################################
def select_road(data_nodes, data_edges, motorway):
        '''
        고속도로 도시도로 선택 함수
        - OSM에서 도시별로 어느 정도 비슷하지만, 
          상이한 도로 라벨을 가지고 있기 때문에 다른 도시에 적용할 때는 확인 필요
          고속도로 : motorway, motorway_link, rest_area, services, trunk, trunk_link
          도시도로 : 그 외
        '''
        #도로 제거 필터(제거) 후 사용하지 않는 nodes 또한 제거 
        def remove_no_use_node(data_edges, data_nodes):
            return data_nodes.loc[set(np.array(data_edges.index.values.tolist())[:,:2].reshape(-1,))]
        
        data_edges["idx"] = range(len(data_edges))
        city_road_edges = copy.deepcopy(data_edges)
        ##city_road_edges filter 
        city_road_edges = city_road_edges.loc[list(map(lambda data: not("motorway" in data), city_road_edges.highway))] #고속도로
        city_road_edges = city_road_edges.loc[list(map(lambda data: not("motorway_link" in data), city_road_edges.highway))] #고속도로 교차로
        city_road_edges = city_road_edges.loc[list(map(lambda data: not("rest_area" in data), city_road_edges.highway))] #고속도로 졸음쉼터
        city_road_edges = city_road_edges.loc[list(map(lambda data: not("services" in data), city_road_edges.highway))] #고속도로 휴계소
        city_road_edges = city_road_edges.loc[list(map(lambda data: not("trunk" in data), city_road_edges.highway))] #천변 도시 고속도로
        city_road_edges = city_road_edges.loc[list(map(lambda data: not("trunk_link" in data), city_road_edges.highway))] #천변 도시 고속도로 교차로
        city_road_edges_mask = set(city_road_edges["idx"].values)
        city_road_edges = data_edges.iloc[list(city_road_edges_mask)] #city_road_edges  
        
        ##city_road filter
        motorway_road_edges_mask = set(range(len(data_edges))) - city_road_edges_mask
        motorway_road_edges = data_edges.iloc[list(motorway_road_edges_mask)] #city_road_edges
        
        #고속도로와 도시도로가 서로 연결되어 있는 nodes 정보
        connect_node_inf = set(np.array(city_road_edges.index.tolist())[:,:2].reshape(-1,)) & set(np.array(motorway_road_edges.index.tolist())[:,:2].reshape(-1,))
        #고속도로 edges, nodes 반환
        if motorway == True:
            motorway_road_edges = motorway_road_edges.drop(["idx"], axis=1)
            motorway_nodes = remove_no_use_node(motorway_road_edges, data_nodes)
            
            print(f"- motorway data : rink {len(motorway_road_edges)}개, node : {len(motorway_nodes)} 선택")
            return motorway_nodes, motorway_road_edges, connect_node_inf
        #도시도로 edges, nodes 반환
        else:
            city_road_edges = city_road_edges.drop(["idx"], axis=1)
            city_nodes = remove_no_use_node(city_road_edges, data_nodes)
        
            print(f"- city road : rink {len(city_road_edges)}개, node : {len(city_nodes)}개 선택")
            return city_nodes, city_road_edges, connect_node_inf
        
############################################################################################
def load_graph_from_place(place, motorway=None):
    G1, G2 = graph_from_place(place)

    nodes_1, edges_1 = G1 
    nodes_2, edges_2 = G2

    # 겹치는 도로 제거 및 분석 도로 선택
    edges_1 = remove_overlap_roads(edges_1)
    nodes_1, edges_1, connect_node_inf_1 = select_road(nodes_1, edges_1, motorway)

    edges_2 = remove_overlap_roads(edges_2)
    nodes_2, edges_2, _ = select_road(nodes_2, edges_2, motorway)
    return [nodes_1.reset_index(), edges_1.reset_index(), connect_node_inf_1], [nodes_2.reset_index(), edges_2.reset_index()]
    #return [nodes_1, edges_1, connect_node_inf_1], [nodes_2, edges_2]