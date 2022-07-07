### 0. Packages to use 
import osmnx as ox 
from shapely.geometry import Point, LineString 

import geopandas as gpd
import pandas as pd 
import numpy as np
import itertools 
from tqdm import tqdm 
import warnings 

warnings.filterwarnings('ignore')




def euclid_distance_cal(meter):
    ###유클리드 거리와 실제 거리를 기반으로 1미터당 유클리드 거리 추출
    #점 쌍 사이의 유클리드 거리를 계산
    dis_1 = ox.distance.euclidean_dist_vec(36.367658 , 127.447499, 36.443928, 127.419678)
    #직선거리 계산
    dis_2 = ox.distance.great_circle_vec(36.367658 , 127.447499, 36.443928, 127.419678)

    return dis_1/dis_2 * meter

######################################################################################################################
def generate_new_nodes(data_nodes, data_edges, meter, mode = None, dummy = None): 
    if mode == "cross" or mode == "single_way":
        segment_node = [] #new_nodes
        edge_mask = [] # -> 분할에 사용한 rink를 rink 분할 시 넘겨줘서 살펴보기 분할이 필요없는 모든 rink까지 살펴보는 일 없게 한다.
    else: 
        segment_node = []
    #mode : cross 또는 dummy : dummy_cross 일 경우 끝 점은 교차로로 인식하지 않지 위해 끝점 노드 ID를 만들어 전달한다.
    if mode == "cross" or dummy == "dummy_cross":
        not_cross_node = pd.DataFrame(data_edges[['u','v']].values.reshape(-1))
        not_cross_node = pd.DataFrame(not_cross_node.value_counts(), columns=['cnt']).reset_index()
        not_cross_node.columns = ['osmid', 'cnt']
        not_cross_node = not_cross_node.loc[(not_cross_node['cnt'] == 1)]['osmid'].values.reshape(-1)
    ###data_edges 분할   
    for idx, i in enumerate(data_edges["geometry"]):
        #dummy : dummy_cross일때 아래 기준일 때는 많은 노드를 만들고, 아닐 경우는 교차로 중심으로 노드를 생성 하도록 하는 것
        if dummy == "dummy_cross":
            if data_edges.iloc[idx].length < 180 and "link" in data_edges.iloc[idx].highway:
                mode = None
            else:
                mode = "cross"   
        else: #군집이 할당되있는 경우 패스(mode : cross, single_way)
            if dummy != "dummy":
                if data_edges.iloc[idx].cluster != -1 and mode == "cross":  
                    continue
                elif mode == "cross" and data_edges.iloc[idx].tunnel == 21:
                    continue                                                 
        ###기준으로 분할
        new_node = list(ox.utils_geo.interpolate_points(i, euclid_distance_cal(meter)))
        #생성된 노드가 2개인 경우는 분할 되지 않은 것임으로 2개 이상이 출력되는 경우 실행
        if len(new_node) > 2:
            ###dummy 노드 생성 할 경우(노드만 생성하기 때문에 엣지를 위한 데이터 생성을 하지 않는다.)
            if dummy == "dummy" or dummy == "dummy_cross":
                if len(new_node) == 3:
                    segment_node.append([new_node[1]])
                elif len(new_node) > 3:
                    if mode == "cross":
                        if len(set(data_edges[['u','v']].iloc[idx].values) & set(not_cross_node)) == 1:
                            index_mask = np.where(np.array([[i in data_edges[['u','v']].iloc[idx].tolist()] for i in not_cross_node]).reshape(-1) == True)[0][0]
                            index_mask = np.where(np.array(data_edges[['u','v']].iloc[idx].tolist()) == not_cross_node[index_mask])[0][0]
                            segment_node.append([new_node[-2 if index_mask == 0 else 1]])
                        else:
                            segment_node.append([new_node[1], new_node[-2]])
                    else: 
                        segment_node.append(list(itertools.chain(*[new_node[1:-1]])))
            ###mode : cross, single_way일 경우
            else:
                if len(new_node) == 3:
                    segment_node.append([new_node[1]])
                    edge_mask.append([idx,1,data_edges[['u','v']].iloc[idx].tolist()])
                #생성된 교차가 4개 이상인 경우
                elif len(new_node) > 3: 
                    #교차로 중심으로 분할 하는 경우
                    if mode == "cross":
                        #끝 점과 연결된 교차로 edge인 경우 끝점에서는 node를 생성안함
                        if (len(set(data_edges[['u','v']].iloc[idx].tolist()) - set(not_cross_node)) == 1) \
                            and (len(set(data_edges[['u','v']].iloc[idx].tolist())) == 2):
                            index_mask = np.where(np.array([[i in data_edges[['u','v']].iloc[idx].tolist()] for i in not_cross_node]).reshape(-1) == True)[0][0]
                            index_mask = np.where(data_edges[['u','v']].iloc[idx].values == not_cross_node[index_mask])[0][0]
                            segment_node.append([new_node[-2 if index_mask == 0 else 1]])
                            edge_mask.append([idx,1,data_edges[['u','v']].iloc[idx].tolist()])
                        #모든 노드와 연결되지 않은 경우 넘어간다.
                        elif len(set(data_edges[['u','v']].iloc[idx].tolist()) - set(not_cross_node)) == 0:
                            continue
                        #위와 같은 조건에 만족하지 않는 경우 그냥 양 끝점의 노드를 생성한다.
                        else:
                            segment_node.append([new_node[1], new_node[-2]])
                            edge_mask.append([idx,2,data_edges[['u','v']].iloc[idx].tolist()])
                    #교차로가 아닌 nodes를 생성할때는 linestring의 시작과 끝점인 0과 -1를 뺀 모든 node를 생성한다.
                    else: 
                        segment_node.extend([new_node[1:-1]])   
                        edge_mask.append([idx, len(new_node[1:-1]), data_edges[['u','v']].iloc[idx].tolist()])                    
            
    #segment_node 이중 리스트 풀어주기
    segment_node = list(itertools.chain(*segment_node)) 
    #x, y (columns) 만들기
    segment_xy = list(map(lambda data: [data[1], data[0]], segment_node))
    segment_nodes = gpd.GeoDataFrame(segment_xy, columns=["y","x"])
    #geometry (columns) 만들기
    segment_nodes["geometry"] = list(map(lambda data: Point(data), segment_node))  
    #state : 3 교차로를 나눠주는 노드/ state : 4 단일로를 나눠주는 node/ state: 999 dummy 노드  
    if dummy == "dummy" or dummy == "dummy_cross":
        segment_nodes["state"] = 999
    else:
        mask = list(map(lambda data: data[0], edge_mask))
        segment_nodes["state"] = 3 if mode == "cross" else 4  
        #edge id (u,v,key, osmid)
        segment_edge_id = list(map(lambda data:[data[2]]* data[1], edge_mask))
        segment_nodes["edge_id"] = list(itertools.chain(*segment_edge_id))
    #segment_nodes_data. index 설정
    
    segment_nodes['osmid'] = [i for i in range(data_nodes['osmid'].max()+1, data_nodes['osmid'].max()+len(segment_nodes)+1)]
    data_nodes = pd.concat([data_nodes, segment_nodes])
    if dummy == "dummy" or dummy == "dummy_cross":
        return data_nodes
    else:
        return data_nodes, mask    
    
################################################################################################################
def Line_split(line_string, point):
    line_string = line_string.geometry.values[0]
    point = point.geometry.values[0]
    #point가 linestring 위에 있는지 확인
    assert point.distance(line_string) < 1e-12, "point가 linestring 위에 있지 않습니다!!"
    #linestring split
    if len(line_string.coords) == 2:
        Line1,Line2 = LineString([Point(line_string.coords[0]), point]),\
            LineString([point, Point(line_string.coords[1])])
        return Line1, Line2
    else: 
        check_num = None
        for i in range(len(line_string.coords)-1):
            check_Line = LineString([Point(line_string.coords[i]),Point(line_string.coords[i+1])])
            if point.distance(check_Line) < 1e-10: check_num = i
        L1_point = [Point(line_string.coords[i]) for i in range(0, check_num+1)]
        L2_point = [Point(line_string.coords[i]) for i in range(check_num+1, len(line_string.coords))]
        L1_point.append(point)
        L2_point.insert(0, point)
        Line1,Line2 = LineString(L1_point),LineString(L2_point)
        return Line1, Line2

#select data function
def select_data(data, point, index, tunnel=False):
    MASK = [point.geometry.iloc[0].distance(data.geometry.iloc[i]) <1e-10 for i  in range(len(data))]
    using_data_1 = data.iloc[[not i for i  in MASK]]
    deleted_data_1 = data.iloc[MASK]
    if tunnel:
        return using_data_1, deleted_data_1
    if len(deleted_data_1) > 1:               
        using_data_2 = deleted_data_1.loc[[index != i for i in deleted_data_1[['u','v','key']].values.tolist()]]
        deleted_data_2 = deleted_data_1.loc[[index == i for i in deleted_data_1[['u','v','key']].values.tolist()]]
        using_data = pd.concat([using_data_1, using_data_2])
        return using_data, deleted_data_2
    return using_data_1, deleted_data_1

#새로운 linestring 길이 구하는 함수(단위:m)
def linstring_length(linestring):
    linestring = linestring.coords
    return sum([ox.distance.great_circle_vec(linestring[i][1],linestring[i][0],linestring[i+1][1],linestring[i+1][0]) for i in range(len(linestring)-1)])


#노드 찾는 함수        
def find_node(data_nodes, data_edge_target):
    data_nodes.index = data_nodes['osmid']
    
    u_list = []
    v_list = []
    key_list = []
    for i in tqdm(range(len(data_edge_target))):
        data_edge_target_uv = data_edge_target.iloc[[i]]
        data_edge_target_uv["geometry"] = Point(data_edge_target.geometry.values[i].coords[0])#U
        u = data_nodes[["geometry"]].sjoin(data_edge_target_uv[["geometry"]], how="left", predicate="intersects").dropna(axis=0).index[0]
        data_edge_target_uv["geometry"] = Point(data_edge_target.geometry.values[i].coords[-1]) #V
        v = data_nodes[["geometry"]].sjoin(data_edge_target_uv[["geometry"]], how="left", predicate="intersects").dropna(axis=0).index[0]
        key = 0
        
        u_list.extend([u])
        v_list.extend([v])
        key_list.extend([key])
        
    data_edge_target['u'] = u_list
    data_edge_target['v'] = v_list
    data_edge_target['key'] = key_list

    return data_edge_target 
        
### Step6 - Create edges based on conditions            
def generate_new_edges(data_nodes,data_edges, mask, mode = None , tunnel = False):
    #for 문에 많은 반복을 줄이기 위해 분할될 edges/ 분할되지 않을 edges를 구분해준다!
    if tunnel:
        data_edge_target = data_edges.loc[[not i for i in data_edges.tunnel.isnull()]]
        data_edge_non_target = data_edges.loc[data_edges.tunnel.isnull()]
    else:
        data_edge_target = data_edges.iloc[sorted(mask)]
        data_edge_non_target = data_edges.loc[[not(i in mask) for i in range(len(data_edges))]]

    #교차로 중심 분할일 때는 state가 3으로 정의된 노드를 중심으로 분할
    if mode == "cross":
        data_nodes_target = data_nodes.loc[data_nodes["state"] == 3]
    #터널 노드로 분할할 시 
    elif tunnel:
        data_nodes_target = data_nodes.loc[[i in set(set(data_nodes.loc[data_nodes.state == 2]['osmid']) - mask) for i in data_nodes['osmid']]]
    #단일로 분할일 때는 state가 4로 정의된 노드를 중심을 분할
    else:
        data_nodes_target = data_nodes.loc[data_nodes["state"] == 4]
    for i in range(len(data_nodes_target)):
        new_node = data_nodes_target.iloc[[i]]
        if tunnel:
            data_edge_target, deleted_data = select_data(data_edge_target, new_node, None, tunnel)
        else:
            UVKEY = new_node["edge_id"].values[0]
            data_edge_target, deleted_data = select_data(data_edge_target, new_node, UVKEY)
        Line1, Line2 = Line_split(deleted_data,new_node)
        for line in [Line1, Line2]:
            #u, v, key
            u = deleted_data['u']
            v = deleted_data['v']
            key = 0
            
            #u, v, key, osmid, oneway, lanes, name, highway, maxspeed, length, geometry
            features = deleted_data.iloc[[0]].values[0]
            features[0] = u
            features[1] = v
            features[2] = key
            features[8], features[9] = linstring_length(line), line
            features = features.tolist()
            
            added_edge_data = gpd.GeoDataFrame(np.array([features]), columns= deleted_data.columns)
            data_edge_target = gpd.GeoDataFrame(pd.concat([data_edge_target, added_edge_data]))
                  
    data_edge_target = find_node(data_nodes, data_edge_target)          
    data_edges = gpd.GeoDataFrame(pd.concat([data_edge_target, data_edge_non_target]))
    if not(tunnel):
        data_nodes = data_nodes.drop(["edge_id"],axis=1)
    return data_nodes.reset_index(drop=True), data_edges.reset_index(drop=True)


####################################################################################################
def generate_tunnel_edges(main_road, non_main_road):
    '''
    터널 노드로 edge 업데이트
    '''
    main_road_nodes, main_road_edges = generate_new_edges(main_road[0], main_road[1], mask = main_road[2], tunnel = True)
    non_main_road_nodes, non_main_road_edges = generate_new_edges(non_main_road[0], non_main_road[1], mask = non_main_road[2], tunnel=True)
    
    return [main_road_nodes, main_road_edges], [non_main_road_nodes, non_main_road_edges]

def generate_cross_nodes_edeges(data_nodes, data_edges, meter=150): 
    '''
    교차로 150m 단위로 분할(길이 기준은 사용자가 바꿔도 됨)
    '''
    data_nodes, mask = generate_new_nodes(data_nodes, data_edges, meter = meter, mode = "cross")
    data_nodes, data_edges = generate_new_edges(data_nodes, data_edges, mask = mask, mode = "cross") 
    
    return data_nodes, data_edges

def generate_singleway_nodes_edeges(data_nodes, data_edges, meter=600): 
    '''
    단일로 600m 단위로 분할(길이 기준은 사용자가 바꿔도 됨)
    '''
    data_nodes, mask = generate_new_nodes(data_nodes, data_edges, meter = meter, mode = "single_way")
    data_nodes, data_edges = generate_new_edges(data_nodes, data_edges, mask = mask, mode = "single_way") 
    
    return data_nodes, data_edges