from generator_node_edge import generate_new_nodes, euclid_distance_cal

from sklearn.cluster import DBSCAN
import pandas as pd 
import warnings 

warnings.filterwarnings('ignore')



def generate_cluster(data_nodes, cluster_meter, min_sample = 4):
    
    X_main_road = data_nodes[["y","x"]]

    main_road_dbscan = DBSCAN(eps=euclid_distance_cal(cluster_meter), min_samples=min_sample)
    main_road_cluster = main_road_dbscan.fit_predict(X_main_road)    
    data_nodes["cluster"] = main_road_cluster
    return data_nodes 

def redefine_cluster_id(data_nodes, data_edges, mode = None):
    #outlier인 군집은 다시 -1로 전환
    if mode == "cross":
        outlier_mask =  [[id,-1] if i == 1 else [id,id] for id, i in zip(data_nodes.cluster.value_counts().index,data_nodes.cluster.value_counts())]
        outlier_dict = dict()
        for i in outlier_mask:
            outlier_dict[i[0]] = i[1]
        outlier_mask = outlier_dict
        data_nodes.cluster = [outlier_mask[i] if i else i for i in data_nodes.cluster]
        #군집 번호 다시 한번 다시 정의
        outlier_dict = dict()
        for i, j in zip(sorted(set(data_nodes.cluster)),range(-1,len(set(data_nodes.cluster))-1)):
            outlier_dict[i] = i
        outlier_mask = outlier_dict
        data_nodes.cluster = [outlier_mask[i] for i in data_nodes.cluster]   
    else:
        #tunnel 일 경우 20000 부터 시작
        data_nodes.tunnel_cluster = data_nodes.cluster.values + 20000
        data_nodes.cluster = data_nodes.cluster.values + 20000
            

    #data_edges 군집 번호 부여
    if not "cluster" in data_edges.columns:
        data_edges["cluster"] = 0
    cluster_mask = dict()
    for id, value in zip(data_nodes['osmid'].values , data_nodes.cluster.values):
        cluster_mask[id] = value

    data_edges["cluster"] = [i[0] if i[0] == i[1] else -1 for i in [[cluster_mask[i[0]],cluster_mask[i[1]]] for i in data_edges[['u','v']].values]]
    return data_nodes, data_edges


def tunnel_data_prepared(data_nodes, data_edges):
    # 터널 상태 0으로 만들기
    data_edges.tunnel = 0 
    #tunnel nodes id 추출
    tunnel_node_idx = set(data_nodes.loc[data_nodes.state == 2]['osmid'])
    
    #edges : tunnel이면 1 아니면 0 대입 
    data_edges.tunnel = [1 if i else 0 for i in list(map(lambda data: (data[0] in tunnel_node_idx) and (data[1] in tunnel_node_idx), data_edges[['u','v']].values.tolist()))]
    return data_nodes, data_edges

def split_tunnel_data(data_nodes, data_edges):
    tunnel_edges = data_edges.loc[data_edges.tunnel == 1]
    tunnel_nodes = data_nodes.loc[data_nodes.state == 2]
    non_tunnel_edges = data_edges.loc[data_edges.tunnel != 1]
    non_tunnel_nodes = data_nodes.loc[data_nodes.state != 2]
    return tunnel_nodes, tunnel_edges, non_tunnel_nodes, non_tunnel_edges

def split_cross_data(data_nodes, data_edges):
    cross_edges = data_edges.loc[data_edges.tunnel != 21]
    non_cross_edges = data_edges.loc[data_edges.tunnel == 21]
    
    cross_node_mask = set(cross_edges[['u','v']].values.reshape(-1))
    cross_nodes = data_nodes.loc[[i in cross_node_mask for i in data_nodes['osmid']]]
    non_cross_nodes = data_nodes.loc[[i in set(set(data_edges[['u','v']].values.reshape(-1)) - cross_node_mask) for i in data_nodes['osmid']]]
    return cross_nodes, cross_edges, non_cross_nodes, non_cross_edges 



###################################################################################################################
'''
- 터널 종류 분류
    - 간선도로
        - 교차로 내 터널 state : 21
        - 도로 연장 터널 state : 22
    - 집산 도로   
        - 터널, 터널 state : 2
'''

def cluster_for_tunnel_1(data_nodes, data_edges):
    '''
    단순 터널 분류
    '''
    data_nodes, data_edges = tunnel_data_prepared(data_nodes, data_edges)
    tunnel_nodes, tunnel_edges, non_tunnel_nodes, non_tunnel_edges = split_tunnel_data(data_nodes, data_edges)
    tunnel_dummy = generate_new_nodes(tunnel_nodes, tunnel_edges, meter = 20, dummy = "dummy")
    tunnel_dummy = generate_cluster(tunnel_dummy, cluster_meter=50, min_sample = 2)
    assert sum(tunnel_dummy.cluster == -1) == 0, "터널 군집이 제대로 되지 않았습니다."
    tunnel_nodes = tunnel_dummy.loc[tunnel_dummy.state != 999]
    tunnel_nodes, tunnel_edges = redefine_cluster_id(tunnel_nodes, tunnel_edges)

    data_nodes = pd.concat([non_tunnel_nodes, tunnel_nodes])
    data_edges = pd.concat([non_tunnel_edges, tunnel_edges])
    return data_nodes, data_edges


def cluster_for_tunnel_2(data_nodes, data_edges):
    '''
    간선도로의 교차로 내 터널, 도로 연장 성격의 터널 분류 
    '''
    
    tunnel_nodes, tunnel_edges, non_tunnel_nodes, non_tunnel_edges = split_tunnel_data(data_nodes, data_edges)
    tunnel_edges["x"] = tunnel_edges.geometry.centroid.x 
    tunnel_edges["y"] = tunnel_edges.geometry.centroid.y
    geometry = tunnel_edges["geometry"].values
    tunnel_edges["geometry"] = tunnel_edges.geometry.centroid 

    # 
    x = tunnel_edges[["y","x","geometry"]]
    y = non_tunnel_nodes[["y","x","geometry"]]

    x['inf'] = 'edges' 
    y['inf'] = 'nodes'

    tunnel_edges_nodes = pd.concat([x, y])

    tunnel_edges["geometry"] = geometry

    tunnel_nodes_cluster = generate_cluster(tunnel_edges_nodes, 30, min_sample = 4)
    tunnel_nodes_cluster = tunnel_nodes_cluster.loc[[i == 'edges' for i in tunnel_nodes_cluster.inf]]
    tunnel_cluster_inf = pd.DataFrame(tunnel_nodes_cluster.cluster.value_counts())
    mask = tunnel_cluster_inf.loc[tunnel_cluster_inf.cluster == 2].index 

    mask_21 = tunnel_nodes_cluster.loc[list(map(lambda data: data in mask ,tunnel_nodes_cluster.cluster))][['x','y']].values.tolist()
    mask_21 = [i in mask_21 for i in tunnel_edges[['x','y']].values.tolist()]
    tunnel_edges.tunnel = [21 if i else 22 for i in mask_21]

    tunnel_edges = tunnel_edges.drop(["x","y"],axis=1)

    tunnel_state = dict()
    for i,j in zip(tunnel_edges[['u','v']].values,tunnel_edges.tunnel):
        tunnel_state[i[0]] = j 
        tunnel_state[i[1]] = j

    now_state = []
    for i,j in zip(tunnel_nodes['osmid'].values, tunnel_nodes.state):
        try:
            now_state.append(tunnel_state[i])
        except:
            now_state.append(j) 
            
    tunnel_nodes['state'] = now_state
    data_nodes = pd.concat([non_tunnel_nodes, tunnel_nodes])        
    data_edges = pd.concat([non_tunnel_edges, tunnel_edges])
    return data_nodes, data_edges 


def cluster_for_cross(data_nodes, data_edges, split_meter=30, cluster_meter= 55, min_sample=4):
    cross_nodes, cross_edges, non_cross_nodes, non_cross_edges = split_cross_data(data_nodes, data_edges)
    cross_dummy = generate_new_nodes(cross_nodes, cross_edges, meter=split_meter, dummy = "dummy_cross")
    cross_dummy = generate_cluster(cross_dummy, cluster_meter=cluster_meter, min_sample=min_sample)
    cross_nodes = cross_dummy.loc[cross_dummy.state != 999]
    cross_nodes, cross_edges = redefine_cluster_id(cross_nodes, cross_edges, mode='cross')
    
    data_nodes = pd.concat([cross_nodes, non_cross_nodes])
    data_edges = pd.concat([cross_edges, non_cross_edges])
    return data_nodes, data_edges