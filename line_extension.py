### 0. Packages to use 
from shapely.geometry import Point, MultiLineString 
from shapely import ops 

import geopandas as gpd
import pandas as pd 
import numpy as np
import warnings 

warnings.filterwarnings('ignore')

def split_data_for_connect_1(data_edges, connect_node_inf):
    '''
    linestring 연장이 필요한 edges, 필요없는 edges 분리 
    (도로 종류 분리 시 사용 |그냥 연장 시 pass)
    '''
    # connect mask 
    need_connect_mask = list(map(lambda data: data[0] in connect_node_inf or data[1] in connect_node_inf, data_edges[["u","v"]].values))

    # 연결이 필요한 edge, 필요없는 edge 분리
    need_connect_edge = data_edges.loc[need_connect_mask]
    need_non_connect_edge = data_edges.loc[[not i for i in need_connect_mask]]
    return [need_connect_edge, need_non_connect_edge]   

def split_data_for_connect_2(data, simple_extension=False):
    ##연결 정보 생성
    data_nodes, data_edges, connect_node_inf  = data

    if simple_extension:
        need_connect_edge = data_edges #전체가 연장 타겟
        need_non_connect_edge = need_connect_edge.iloc[[0]].drop(index=[need_connect_edge.index[0]])
        need_connect_node = data_nodes.loc[[i in set(need_connect_edge[["u","v"]].values.reshape(-1)) for i in data_nodes['osmid']]]           
    else:
        need_connect_edge, need_non_connect_edge = split_data_for_connect_1(data_edges, connect_node_inf) 
        need_connect_node = data_nodes.loc[[i in set(need_connect_edge[["u","v"]].values.reshape(-1)) for i in data_nodes['osmid']]]
    return [need_connect_edge, need_non_connect_edge], need_connect_node


def graph_inf(data_edges):
    '''
    edge data의 nodes 연결 정보를 생성
    '''  

    #그래프 list를 인덱싱하기 위한 dictionary 생성 
    node_osmid_dict = dict()
    for idx, i in enumerate(sorted(set(data_edges[['u','v']].values.reshape(-1)))):
        node_osmid_dict[i] = idx
        
    #nodes, edges index 변수 생성
    nodes_index = sorted(set(data_edges[['u','v']].values.reshape(-1)))
    edges_index = data_edges[['u','v']].values.tolist()

    #연결 정보 생성
    adj = [[] for _ in range(len(nodes_index))]

    for src, dst in edges_index:
        adj[node_osmid_dict[src]].append(dst)
        adj[node_osmid_dict[dst]].append(src)
        
    #연결 정보 dictionary로 표출   
    graph = dict()
    for node_idx, connect_list in zip(nodes_index, adj):
        graph[node_idx] = connect_list
    return graph

def check_node_state(data_edges, connect_node_inf, simple_extension):
    '''
    연장에 사용하는 node가 끝점, 중간 점 node ID 추출 
    (사용 : middle,start_to_end, 미사용 : all/debugging을 위해 출력)
    '''
    nodes_inf = pd.DataFrame(pd.DataFrame(data_edges[['u','v']].values.reshape(-1)).value_counts(), columns=['cnt'])
    ###연결에 사용되는 모든 nodes
    all = set(np.array(nodes_inf.index.tolist()).reshape(-1))
    if simple_extension:
        #단순한 연결에 사용 (끝점과 교차로만 빼고 나머지 연결)
        start_to_end = set(np.array(nodes_inf.loc[list(np.array(nodes_inf['cnt'] == 1) + np.array(nodes_inf['cnt'] > 2))].index.tolist()).reshape(-1)) \
            | connect_node_inf
    else:
        ###시작이자 끝 점인 nodes = 끝 점 + 교차로 + 도로 필수 노드 정보 
        start_to_end = set(np.array(nodes_inf.loc[list(np.array(nodes_inf['cnt'] == 1) + np.array(nodes_inf['cnt'] > 2))].index.tolist()).reshape(-1)) \
            | (all-connect_node_inf)
    return start_to_end


###linestring 연결 및 data 통합
def dfs(graph, list_to_exclude, start_node):    
    ## 기본은 항상 두개의 리스트를 별도로 관리해주는 것
    visited, need_visited = list(), list()
    
    need_visited.append(start_node)
    ## 만약 아직도 방문이 필요한 노드가 있다면,
    while need_visited:
        ## 그 중에서 가장 마지막 데이터를 추출 (스택 구조의 활용)
        for i in need_visited:
            check_point = []
            if i not in visited:
                visited.extend([i]) 
                if i in list_to_exclude:
                    node = []
                else:    
                    node = graph.pop(i)
                ## 만약 그 노드가 방문한 목록에 없다면
                for j in node:
                    if j not in visited:
                        ## 방문한 목록에 추가하기 
                        check_point.extend([j])
                        visited.extend([j])
                        if j in list_to_exclude:
                            need_visited.extend([])
                        else:
                            need_visited.extend(graph[j])
        if len(check_point) == 0:
            break            

    for i in visited:
        try:
            graph.pop(i)
        except:
            pass
    return graph, visited  

def connect_point_list(data_nodes, graph, list_to_exclude):
    start_and_end = set(data_nodes["osmid"]) - list_to_exclude
    start_and_end, list_to_exclude = list(start_and_end), list(list_to_exclude)
    
    end_point_list = []
    n = len(graph)
    print("||| Linestring_extraction")
    
    while start_and_end:
        graph, visited = dfs(graph, list_to_exclude, list(graph.keys())[0])
        end_point_list.append(visited)
        for j in visited:
            try:
                start_and_end.remove(j)
            except:
                pass
        #print(f"\t {n-len(graph)}/{n}")        
    while graph:
        if len(graph) > 0:
            graph, visited = dfs(graph, list_to_exclude, list(graph.keys())[0])
            end_point_list.append(visited)
            for j in visited:
                try:
                    start_and_end.remove(j)
                except:
                    pass
        #print(f" : {n-len(graph)}/{n}")
    return end_point_list, list_to_exclude

def connect_check(data, mask, list_to_exclude):
    if data[0] in list_to_exclude and data[1] in list_to_exclude:
        return 9999
    else: 
        if data[0] in list_to_exclude: 
            return mask[data[1]]
        elif data[1] in list_to_exclude: 
            return mask[data[0]]
        else: 
            return mask[data[0]]
        
def connect_inf(data_nodes, data_edges, graph, list_to_exclude):
    end_connect_inf, list_to_exclude = connect_point_list(data_nodes, graph, list_to_exclude)
    MC = dict()
    for idx, i in enumerate(end_connect_inf):
        MC[idx] = i
        
    mc = dict()
    for k, v in zip(MC.keys(), MC.values()):
        for i in v:
            mc[i] = k
    
    # 연결 정보 변수 생성
    data_edges["cn_inf"] = list(map(lambda data: connect_check(data, mc, list_to_exclude), data_edges[['u','v']].values.tolist()))
    return data_edges      



#######edges data 정보 통합
def merge_inf(need_connect_node, need_connect_edge, need_non_connect_edge): 
    def merge_information(data_edge_subset, data_nodes):
        #정보 합치기 (list 일때와 아닐 때 구분)
        def check_in_list(data):
            list_data = []
            for i in data:
                try:
                    if np.isnan(i):
                        pass
                except:
                    if type(i) == list:
                        list_data.extend(i)
                    else:
                        list_data.append(i)   
                if len(list_data) == 0:
                    list_data.append(np.nan)
            return list_data
        
        #합쳐진 target edge 저장할 빈 DataFrame
        data_edge_target = data_edge_subset.iloc[[0]].drop(index=[data_edge_subset.index[0]])

        if len(data_edge_subset) > 1:
            
            feature = []
            
            for i in data_edge_subset.columns:
                if i == "geometry":
                    multi_line = MultiLineString(data_edge_subset[i].values)
                    geometry = ops.linemerge(multi_line) 
                    feature.extend([geometry]) 
                    
                elif i == "length":
                    length = sum(data_edge_subset[i])
                    feature.extend([length])
                    
                elif (i == 'u') | (i == 'v') | (i == 'key'):
                    pass
                
                elif len(set(check_in_list(data_edge_subset[i]))) > 1:
                    globals()[f"{i}"] = list(set(check_in_list(data_edge_subset[i])))
                    feature.extend([globals()[f"{i}"]])
                else:
                    globals()[f"{i}"] = list(set(check_in_list(data_edge_subset[i])))[0]
                    feature.extend([globals()[f"{i}"]])       
                    
            node_info = set(data_edge_subset[['u','v']].values.reshape(-1))
            nodes_subset = data_nodes.loc[[i in node_info for i in data_nodes['osmid']]]

            u = nodes_subset.loc[[i == Point(geometry.coords[0]) for i in nodes_subset.geometry.values]]['osmid'].values[0]
            v = nodes_subset.loc[[i == Point(geometry.coords[-1]) for i in nodes_subset.geometry.values]]['osmid'].values[0]
            key = 0      
            
            id_feature = [u,v,key]
            id_feature.extend(feature)
            
            connect_edge_data = gpd.GeoDataFrame(np.array([id_feature]), columns= data_edge_subset.columns)
            data_edge_target = gpd.GeoDataFrame(pd.concat([data_edge_target, connect_edge_data]))
        else:
            data_edge_target = gpd.GeoDataFrame(pd.concat([data_edge_target, data_edge_subset]))
        return data_edge_target    
    
    for i in set(need_connect_edge.cn_inf):
        need_connect_edge_subset = need_connect_edge.loc[need_connect_edge.cn_inf == i]
        if i != 9999:
            merge_edge_data = merge_information(need_connect_edge_subset, need_connect_node)
        else:
            merge_edge_data = need_connect_edge_subset
        need_non_connect_edge = pd.concat([need_non_connect_edge, merge_edge_data])
    need_non_connect_edge = need_non_connect_edge.drop(["cn_inf"], axis=1)
    return need_non_connect_edge

############################################################################################################################
def linestring_extension(data, simple_extension=False):
    
    data_nodes, data_edges, connect_node_inf = data
    
    if simple_extension:
        need_connect_edge = data_edges #전체가 연장 타겟
        need_non_connect_edge = need_connect_edge.iloc[[0]].drop(index=[need_connect_edge.index[0]]) #dummy  
        need_connect_node = data_nodes.loc[[i in set(need_connect_edge[['u','v']].values.reshape(-1)) for i in data_nodes['osmid']]]           
    else:
        edges, nodes = split_data_for_connect_2(data)
        need_connect_edge, need_non_connect_edge = edges
        need_connect_node = nodes 

    graph = graph_inf(need_connect_edge)
    list_to_exclude = check_node_state(need_connect_edge, connect_node_inf, simple_extension)
    need_connect_edge = connect_inf(need_connect_node, need_connect_edge, graph, list_to_exclude)
    edges = merge_inf(need_connect_node, need_connect_edge, need_non_connect_edge)
    mask = set(data_nodes['osmid']) & set(edges[['u','v']].values.reshape(-1))
    nodes = data_nodes.loc[[i in mask for i in data_nodes['osmid']]]
    return nodes.reset_index(drop=True), edges.reset_index(drop=True)

# input : G2 
def tunnel_road_preprogress(data):
    G2_nodes, G2_edges = data

    tunnel_edges = G2_edges.loc[[not i for i in G2_edges.tunnel.isnull()]]
    non_tunnel_edges = G2_edges.loc[G2_edges.tunnel.isnull()]

    tunnel_nodes = G2_nodes.loc[[i in set(tunnel_edges[['u','v']].values.reshape(-1)) for i in G2_nodes['osmid']]]
    tunnel_mask = set(tunnel_edges[['u','v']].values.reshape(-1)) & set(non_tunnel_edges[['u','v']].values.reshape(-1))
    return [tunnel_nodes, tunnel_edges, tunnel_mask]

def tunnel_linestring_extension(data):
    G2 = tunnel_road_preprogress(data)
    if len(G2[-1]) == 0:
        tunnel_nodes, tunnel_edges, _ = G2
    else:
        tunnel_nodes, tunnel_edges = linestring_extension(G2, simple_extension=True)
    return [tunnel_nodes, tunnel_edges]