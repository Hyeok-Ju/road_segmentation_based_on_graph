import pandas as pd 
import numpy as np
import warnings 

warnings.filterwarnings('ignore')



def generate_main_road_cluster(data_nodes, data_edges):
    def nodes_set_cluster_of_based_on_edges(data_nodes, data_edges, mode="base"):
        if mode == "base":
            #edges data 기반 cluster dictionary 생성 (based_on_edges_cluster)
            u_cluster = np.concatenate([data_edges['u'].values.reshape(-1,1), data_edges.cluster.values.reshape(-1,1)],axis=1)
            v_cluster = np.concatenate([data_edges['v'].values.reshape(-1,1), data_edges.cluster.values.reshape(-1,1)],axis=1)
            cluster_mask = np.concatenate([u_cluster, v_cluster], axis=0) 

            based_on_edges_cluster = dict()
            for id, cls in zip(cluster_mask[:,:1],cluster_mask[:,1:2]):
                if id[0] in based_on_edges_cluster.keys() and cls[0] ==  -1:
                    pass
                else:
                    based_on_edges_cluster[id[0]] = cls[0]
            #### 여기 부터   
            ###edges 기반 nodes 군집 재부여 -> 연결되있는 군집만 유지   
            data_nodes.cluster = [based_on_edges_cluster[id] if st != 21 else data_nodes.cluster[id] for id, st in zip(data_nodes['osmid'].values.tolist(), data_nodes.state)]
        else:
            data_nodes.cluster = [-1 if i == 9999 else i for i in data_nodes.cluster]
        return data_nodes

    def advance_cluster(data_nodes, data_edges, step = 1):    
        if step == 1:
            cluster_max_num = max(data_nodes.cluster)

            ###군집 부여 안된 교차로 군집 부여
            regenerate_cluster = list()
            for cls, st in zip(data_nodes.cluster, data_nodes.state):
                if st == 1 and cls == -1:
                    cluster_max_num = cluster_max_num + 1
                    regenerate_cluster.append(cluster_max_num)
                else: 
                    regenerate_cluster.append(cls)
                    
            data_nodes.cluster = regenerate_cluster
            
            cluster_id = dict()
            for idx, cls in zip(data_nodes['osmid'].values.tolist(), data_nodes.cluster):
                if type(cls) != list:
                    cls = int(cls)
                cluster_id[idx] = cls
                
            one_advance_cluster = []
            for idx, cls in zip(data_edges[['u','v']].values.tolist(), data_edges.cluster):
                if cluster_id[idx[0]] != cluster_id[idx[1]]:
                    cls = list(set([cluster_id[idx[0]], cluster_id[idx[1]]]) - set([-1]))
                    if len(cls) ==  1:
                        one_advance_cluster.append(cls[0])
                    else: 
                        one_advance_cluster.append(cls)
                else:
                    one_advance_cluster.append(cls)
                    
            data_edges.cluster = one_advance_cluster
        else:
            cluster_id = dict()
            for idx, cls in zip(data_nodes['osmid'].values, data_nodes.cluster):
                if type(cls) != list:
                    cls = int(cls)
                cluster_id[idx] = cls

            two_advance_cluster = []
            for idx, cls, lng in zip(data_edges[['u','v']].values.tolist(), data_edges.cluster, data_edges["length"]):
                if type(cls) == list or lng > 150:
                    two_advance_cluster.append(cls)
                elif cluster_id[idx[0]] != cluster_id[idx[1]]: 
                    if type(cluster_id[idx[0]]) == list or type(cluster_id[idx[1]]) == list:
                        two_advance_cluster.append(cls)
                    else:
                        if cls == -1:
                            cls = list(set([cluster_id[idx[0]], cluster_id[idx[1]]]) - set([-1]))
                            if len(cls) ==  1:
                                two_advance_cluster.append(cls[0])
                            else: 
                                two_advance_cluster.append(cls)
                        else:
                            two_advance_cluster.append(cls)
                else:
                    two_advance_cluster.append(cls)
                    
            data_edges.cluster = two_advance_cluster    
        return data_nodes, data_edges
    #step 1 
    data_nodes = nodes_set_cluster_of_based_on_edges(data_nodes, data_edges)
    data_nodes, data_edges = advance_cluster(data_nodes, data_edges)
    #step 2
    data_nodes_copy = data_nodes.copy()
    data_nodes = nodes_set_cluster_of_based_on_edges(data_nodes, data_edges)
    data_nodes, data_edges = advance_cluster(data_nodes, data_edges, step = 2)
    data_nodes = data_nodes_copy
    return data_nodes, data_edges


def generate_non_main_road_cluster(data_nodes, data_edges):
    #연결 기초 정보 생성
    def start_point_and_graph_inf(data_nodes, data_edges):    
        #end point 이자 start point인 한 번 언급된 point 추출   
        number_of_nodes_of_edges = pd.DataFrame(pd.DataFrame(data_edges[['u','v']].values.reshape(-1)).value_counts())
        start_end_point = np.array(number_of_nodes_of_edges.loc[number_of_nodes_of_edges.values == 1].index.tolist()).reshape(-1).tolist()
        
        nodes_index = data_nodes['osmid'].values.tolist()
        edges_index = data_edges[['u','v']].values.tolist()
        
        adj = [[] for _ in range(len(data_nodes))]

        for src,dst in edges_index:
            adj[src].append(dst)
            adj[dst].append(src)
        ###graph inf    
        graph = dict()
        for node_idx,connect_list in zip(nodes_index, adj):
            graph[node_idx] = connect_list
        return graph, start_end_point


    ###DFS 알고리즘
    def dfs(graph, start_node, data_nodes):
        ###간선도로 연결 정보
        main_road_connect_nodes = set(data_nodes.loc[data_nodes.connect_inf == 1]['osmid'])
        
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
                    if i in main_road_connect_nodes:
                        node = []
                    else:    
                        node = graph.pop(i)
                    ## 만약 그 노드가 방문한 목록에 없다면
                    for j in node:
                        if j not in visited:
                            ## 방문한 목록에 추가하기 
                            check_point.extend([j])
                            visited.extend([j])
                            if j in main_road_connect_nodes:
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

    ###조건에 따른 그래프 DFS 진행
    def connect_point_list(data_nodes, data_edges):
        graph_inf, start_and_end = start_point_and_graph_inf(data_nodes, data_edges)
        end_point_list = []
        n = len(graph_inf)
                    
        while start_and_end:
            graph_inf, visited = dfs(graph_inf, start_and_end[0], data_nodes)
            end_point_list.append(visited)
            for j in visited:
                try:
                    start_and_end.remove(j)
                except:
                    pass
            print(f"{n-len(graph_inf)}/{n}",end="\r")        
        while graph_inf:
            if len(graph_inf) > 0:
                graph_inf, visited = dfs(graph_inf, list(graph_inf.keys())[0], data_nodes)
                end_point_list.append(visited)
                for j in visited:
                    try:
                        start_and_end.remove(j)
                    except:
                        pass
            print(f"{n-len(graph_inf)}/{n}",end="\r")
        return end_point_list

    ###군집 부여 규칙
    def cluster_check(data_nodes,data,mask):
        main_road_connect_mask = set(data_nodes.loc[data_nodes.connect_inf == 1]['osmid'])
        if data[0] in main_road_connect_mask and data[1] in main_road_connect_mask:
            return 9999
        else: 
            if data[0] in main_road_connect_mask: 
                return mask[data[1]]
            elif data[1] in main_road_connect_mask: 
                return mask[data[0]]
            else: 
                return mask[data[0]]
            
    ###nodes 군집에 따른 edges 군집 부여 
    def generate_connect_inf(data_nodes, data_edges):
        end = connect_point_list(data_nodes, data_edges)

        MC = dict()
        for idx, i in enumerate(end):
            MC[idx] = i
            
        mc = dict()
        for k, v in zip(MC.keys(), MC.values()):
            for i in v:
                mc[i] = k
                
        data_edges["M_category"] = list(map(lambda data: cluster_check(data_nodes, data, mc), data_edges[['u','v']].values.tolist()))
        return data_edges        
    
    data_nodes = data_nodes.sort_index()
    data_edges = generate_connect_inf(data_nodes, data_edges)
    return data_nodes, data_edges