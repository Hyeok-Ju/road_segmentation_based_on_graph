# Custom packages
from graph_loader import load_graph_from_place
from line_extension import linestring_extension, tunnel_linestring_extension
from split_road import split_road_and_change_nodes_state 
from generator_node_edge import generate_tunnel_edges, generate_cross_nodes_edeges, generate_singleway_nodes_edeges
from cluster import cluster_for_tunnel_1, cluster_for_tunnel_2, cluster_for_cross
from cluster_spread import generate_main_road_cluster, generate_non_main_road_cluster
from make_map import generate_main_html, generate_non_main_html

import os 
import warnings 

warnings.filterwarnings('ignore')

### parameter
place = '세종 대한민국'
region =  place.split(' ')[0]

### G1 : 베이스 도시도로, G2 : 자세한 터널을 추가하기 위한 도시도로
G1, G2 = load_graph_from_place(place)

### 도시도로 추출의 위해 고속도로를 제거했기 때문에 고속도로로 인해 나눠진 도로 다시 연결
road_nodes, road_edges = linestring_extension(G1)
tunnel_nodes, tunnel_edges = tunnel_linestring_extension(G2)

### 도시도로 : 간선도로(주간선, 간선, 보조간선), 집산도로로 분리
main_road, non_main_road = split_road_and_change_nodes_state([road_nodes, road_edges], [tunnel_nodes, tunnel_edges])
### G2을 통해 G1에 자세한 터널 위치를 추가하고 nodes, edges 업데이트 
main_road, non_main_road = generate_tunnel_edges(main_road, non_main_road)

### 터널 양방향으로 나눠진 경우 하나의 터널로 군집
main_road_nodes, main_road_edges = cluster_for_tunnel_1(main_road[0], main_road[1])                  # 간선도로 
non_main_road_nodes, non_main_road_edges = cluster_for_tunnel_1(non_main_road[0], non_main_road[1])  # 집산도로 

### 간선도로 - 도로 연장 터널 22, 교차로 터널 분류 21
main_road_nodes, main_road_edges = cluster_for_tunnel_2(main_road_nodes, main_road_edges)

### 간선도로 하나의 교차로에 복잡한 노드가 있는 교차로 하나로 인식하기 위해 군집화
main_road_nodes, main_road_edges = cluster_for_cross(main_road_nodes, main_road_edges)


### cross segement (meter 150) 간선도로
main_road_nodes, main_road_edges = generate_cross_nodes_edeges(main_road_nodes, main_road_edges)   
### single way segment (meter 600) 간선도로
main_road_nodes, main_road_edges = generate_singleway_nodes_edeges(main_road_nodes, main_road_edges) 

### cluster fillna
main_road_nodes.cluster = main_road_nodes.cluster.fillna(9999)
main_road_edges.cluster = main_road_edges.cluster.fillna(9999)
### tunnel fillna 
main_road_edges.tunnel = main_road_edges.tunnel.fillna(0)

### 간선도로 - 150m 공간적 범위 정의
main_road_nodes, main_road_edges = generate_main_road_cluster(main_road_nodes, main_road_edges)

### 집산도로 - 간선도로로 나눠진 집산도로를 블럭단위로 군집화 
non_main_road_nodes, non_main_road_edges = generate_non_main_road_cluster(non_main_road_nodes, non_main_road_edges)
non_main_road_edges.cluster = non_main_road_edges.cluster.fillna(-1)

### cross segement (meter 150) 집산도로
non_main_road_nodes, non_main_road_edges = generate_cross_nodes_edeges(non_main_road_nodes, non_main_road_edges)   
### single way segment (meter 600) 집산도로
non_main_road_nodes, non_main_road_edges = generate_singleway_nodes_edeges(non_main_road_nodes, non_main_road_edges) 

non_main_road_nodes.connect_inf = non_main_road_nodes.connect_inf.fillna(0)
non_main_road_nodes.cluster = non_main_road_nodes.cluster.fillna(-1)

os.chdir('/home/yh_zoo/Graph_segment/result/')
os.mkdir(f'./data/{region}')

main_road_nodes.to_pickle(f'./data/{region}/main_nodes.pkl')
main_road_edges.to_pickle(f'./data/{region}/main_edges.pkl')
non_main_road_nodes.to_pickle(f'./data/{region}/non_main_nodes.pkl')
non_main_road_edges.to_pickle(f'./data/{region}/non_main_edges.pkl')

generate_main_html(main_road_nodes, main_road_edges, place)
generate_non_main_html(non_main_road_edges, place)