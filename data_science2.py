# -*- coding: utf-8 -*-

import networkx as nx
import json
import random
import matplotlib.pyplot as plt 
from networkx.algorithms.community.centrality import girvan_newman
import itertools
import pandas as pd
from networkx import edge_betweenness_centrality as betweenness
from operator import itemgetter


#calculate the GCC for the three graphs
def giant_connected_component(G2005, G2005w, G2006):
  
   #get all the connected compoents then sort them and pick the first one which is GCC
   Gcc1 = sorted(nx.connected_components(G2005), key=len, reverse=True)
   giant_one = G2005.subgraph(Gcc1[0]) 
   #get all the connected compoents then sort them and pick the first one which is GCC
   Gcc2 = sorted(nx.connected_components(G2005w), key=len, reverse=True)
   giant_two = G2005w.subgraph(Gcc2[0])  
   #get all the connected compoents then sort them and pick the first one which is GCC
   Gcc3 = sorted(nx.connected_components(G2006), key=len, reverse=True)
   giant_three = G2006.subgraph(Gcc3[0])
   return giant_one,giant_two,giant_three 


#count the number of edges for the target graph
def count_target_edges():
    file = open('target_graph.txt', 'rb')
    graph = nx.read_edgelist(file, delimiter=',')
    return len(graph.edges())      




#create the three graphs and return their GCC
def create_graphs():
    fileName = "dblp_coauthorship.json"
    data = json.load(open(fileName))
    dblp2005 = nx.Graph()
    dblp2006 = nx.Graph()
    dblp2005w = nx.Graph()
    for tuple_info in data:
        u = tuple_info[0]
        v = tuple_info[1]
        year = tuple_info[2]
        if year == 2005:
            dblp2005.add_edge(u, v)
            #when 2005w has the edge, increment the weight by one
            if dblp2005w.has_edge(u, v):            
                dblp2005w[u][v]['weight'] += 1
            else:
                dblp2005w.add_edge(u, v, weight=1)    
        elif year == 2006:
            dblp2006.add_edge(u, v)
    #report the number of nodes and edges for each graph in a table        
    table_data = [[dblp2005.number_of_nodes(), dblp2005.number_of_edges()],
                     [dblp2006.number_of_nodes(), dblp2006.number_of_edges()],
                     [dblp2005w.number_of_nodes(), dblp2005w.number_of_edges()]]
    table_column = ("Number of Nodes", "Number of Edges")
    table_row = ("dblp2005", "dblp2006","dblp2005w")
    plt.table(cellText=table_data, rowLabels=table_row, rowColours=None, rowLoc='right', colColours=None, colLabels=table_column, loc='center')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("graphs_nodes_edges.pdf")
    plt.show() 
    return giant_connected_component(dblp2005, dblp2005w, dblp2006) 
        
      
                    
      
#calculate the page rank for the three graphs
def calculate_PageRank(G1,G2,G3):
     
    #get the page rank for all nodes then sort them based on the score then pick the top 50
    file = open("dblp2005.pagerank.txt", "w+")  
    dblp2005_top50 = sorted(nx.pagerank(G1).items(), key=lambda x:-x[1])[:50]    
    for x in list(dblp2005_top50):
        file.write(str(x))
        file.write('\n')
    file.close()
    #get the page rank for all nodes then sort them based on the score then pick the top 50
    file2 = open("dblp2006.pagerank.txt", "w+")  
    dblp2006_top50 = sorted(nx.pagerank(G2).items(), key=lambda x:-x[1])[:50]    
    for x in list(dblp2006_top50):
        file2.write(str(x))
        file2.write('\n')
    file2.close()
    #get the page rank for all nodes then sort them based on the score then pick the top 50
    file3 = open("dblp2005w.pagerank.txt", "w+")  
    dblp2005w_top50 = sorted(nx.pagerank(G3).items(), key=lambda x:-x[1])[:50]    
    for x in list(dblp2005w_top50):
        file3.write(str(x))
        file3.write('\n')
    file3.close()

#calculate the edge betweenness for the three graphs
def calculate_Betweenness(G1,G2,G3):

    #get the edge betweenness for all edges then sort them based on the score then pick the top 20
    file = open("dblp2005.betweenness.txt", "w+")  
    dblp2005_top20 = sorted(nx.edge_betweenness_centrality(G1,k=5000).items(), key=lambda x:-x[1])[:20]    
    for x in list(dblp2005_top20):
        file.write(str(x))
        file.write('\n')
    file.close()
    #get the edge betweenness for all edges then sort them based on the score then pick the top 20
    file2 = open("dblp2006.betweenness.txt", "w+")  
    dblp2006_top20 = sorted(nx.edge_betweenness_centrality(G2,k=5000).items(), key=lambda x:-x[1])[:20]   
    for x in list(dblp2006_top20):
        file2.write(str(x))
        file2.write('\n')
    file2.close()
    #get the edge betweenness for all edges then sort them based on the score then pick the top 20
    file3 = open("dblp2005w.betweenness.txt", "w+")  
    dblp2005w_top20 = sorted(nx.edge_betweenness_centrality(G3,k=5000).items(), key=lambda x:-x[1])[:20]    
    for x in list(dblp2005w_top20):
        file3.write(str(x))
        file3.write('\n')
    file3.close()    


#create two core graphs from their orginial graphs         
def create_core_graphs(G1, G2):  
    degree_graph1 = G1.degree()
    #store the nodes with degree >=3
    required_nodes1 = [n for n,v in dict(degree_graph1).items() if v >= 3]
    degree_graph2 = G2.degree()
    #store the nodes with degree >=3
    required_nodes2 = [n for n,v in dict(degree_graph2).items() if v >= 3]
    #from the stored node list with degree >=3, build a subgraph
    return G1.subgraph(required_nodes1), G2.subgraph(required_nodes2)

#calculate friend to friend edge list
def fof(G):   
    #get a node from graph then get its neighbor then get the neighbor node then get its neighbor 
    FOF_graph = nx.Graph()
    for node1 in G.nodes():
        for node2 in G.neighbors(node1):
            for node3 in G.neighbors(node2):
                #check that the node1 is not the same as node2 and node3. Check node3 is not a neighbor of node1 to make
                #sure there is no loop 
                if node1 != node3 and node2 != node3 and node3 not in G.neighbors(node1):
                    FOF_graph.add_edge(node1,node3)
    file = open("FOF_edges.txt", "wb+")
    nx.write_edgelist(FOF_graph, file, delimiter=',')
    file.close()
    print(len(FOF_graph.edges))
    #987973      


#get the target edges     
def target_edges(G1, G2):
    target_graph = nx.Graph()
    #loop over edges in graph2    
    for edges in G2.edges():
        #if it is not included in graph1 then add it to target graph
         if not G1.has_edge(*edges):
            target_graph.add_edge(*edges)
    file = open("target_graph.txt", "wb+")
    nx.write_weighted_edgelist(target_graph, file, delimiter=',')
    file.close()    
    print(len(target_graph.edges))        
    #252968        
           

#random predictor link prediction 
def random_predictor():
    graph_rand = nx.Graph()
    file = open("FOF_edges.txt", "rb")
    graph_fof = nx.read_edgelist(file, delimiter=',')
    fof_edges = list(graph_fof.edges())
    #loop |T| times and add random edge to a graph for random predictor
    for x in range(252968):
        random_edge = random.choice(fof_edges) 
        graph_rand.add_edge(*random_edge)
    file = open("random_predictor.txt", "wb+")
    nx.write_edgelist(graph_rand, file, delimiter=',')
    file.close()  
    print(len(graph_rand.edges))  
    #223122 - different in each run
    
    
 #common neighbors link prediction   
def common_neighbors(G):
    graph_common_neighbors = nx.Graph()    
    file = open("FOF_edges.txt", "rb")
    fof_graph = nx.read_edgelist(file, delimiter=',')
    #get an edge from Friend to Friend list
    for e in fof_graph.edges():
        #calculate the common_neigbors of edge e in graph parameter G
        graph_common_neighbors.add_edge(*e, score=sum(1 for n in nx.common_neighbors(G, *e)))
    file2 = open("graph_common_neighbors.txt", "wb+")       
    nx.write_edgelist(graph_common_neighbors, file2, delimiter=',')
    print(len(graph_common_neighbors.edges))  
    #987973 - FOF 


#Jaccard Coefficient link prediction
def jaccard_coefficient(G):      
    graph_jaccard = nx.Graph()
    file = open("FOF_edges.txt", "rb")
    fof_graph = nx.read_edgelist(file, delimiter=',')
    #assign the JC for G based on freind to friend edges and then add them iterator
    cal = nx.jaccard_coefficient(G, ebunch=fof_graph.edges())
    #from JC iterator create a graph    
    for u, v, x in cal:
            graph_jaccard.add_edge(u, v, score=x)
    file2 = open("graph_jaccard.txt", "wb+")     
    nx.write_edgelist(graph_jaccard, file2, delimiter=',')
    print(len(graph_jaccard))
    #77063

#preferential attachment link prediction
def preferential_attachment(G):
    graph_preferential_attachment = nx.Graph()
    file = open("FOF_edges.txt", "rb")    
    fof_graph = nx.read_edgelist(file, delimiter=',')
    #assign the PA for G based on freind to friend edges and then add them iterator
    cal = nx.preferential_attachment(G, ebunch=fof_graph.edges())
     #from PA iterator create a graph 
    for u, v, x in cal:
            graph_preferential_attachment.add_edge(u, v, score=x)           
    file2 = open("graph_preferential_attachment.txt", "wb+")       
    nx.write_edgelist(graph_preferential_attachment, file2, delimiter=',')
    print(len(graph_preferential_attachment))
    #77063
 

#adamic_adar link prediction
def adamic_adar(G):     
    graph_adamic_adar = nx.Graph()
    file = open("FOF_edges.txt", "rb")    
    fof_graph = nx.read_edgelist(file, delimiter=',')
    #assign the AA for G based on freind to friend edges and then add them iterator
    cal = nx.adamic_adar_index(G, ebunch=fof_graph.edges())
    #from AA iterator create a graph 
    for u, v, x in cal:
        graph_adamic_adar.add_edge(u, v, score=x)
    file2 = open("graph_adamic_adar.txt", "wb+")     
    nx.write_edgelist(graph_adamic_adar, file2, delimiter=',')
    print(len(graph_adamic_adar))
    #77063    
        
        
#method to calculate the precision and accuracy of different link prediction methods       
def precision(filename, k, resultFile):
        file = open(filename, "rb") 
        precision_graph = nx.read_edgelist(file, delimiter=',')
        #get the top K edges based on the score
        top = sorted(nx.get_edge_attributes(precision_graph, 'score').items(), key=lambda x:-x[1])[:k]
        
 
        #get the target edges to use them for testing the precision
        file2 = open("target_graph.txt", "rb")    
        target_graph = nx.read_edgelist(file2, delimiter=',')

        #if the edge from a graph that was created by specific link prediction method is included inside
        #the target edge list then increment predicte by one 
        predicte = 0
        for x in top:
            if target_graph.has_edge(*x[0]):
                predicte += 1
        file3 = open(resultFile, "w+")     
        #devide the predicted result by K get the precision
        file3.write("precision of " + filename +" for k = "+ str(k) + " is " + str(predicte/k) )
        print(len(target_graph))
        #000  
            
#get the 10 communities based on girvan_newman_modified algorithim         
def community_detection_in_graph(G):
    
    
    file_betweenness = open("dblp2005.betweenness.txt", "rb")    
    graph_betwenness = nx.read_weighted_edgelist(file_betweenness, delimiter=',' )
       
    community_Sizes = []
    k = 10    
    comp = girvan_newman_modified(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    #loop over the all communities 
    for communities in limited:
        #if there is 10 communities in specific stage of algorithm 
        if len(communities) == k:    
            print(communities)
            #get get the size of 10 communities
            for community in communities:
                print(len(community))
                community_Sizes.append(len(community))
    file = open("size_communities.txt", "w+")
    file.write(str(sorted(community_Sizes, reverse=True)))
    file.close()            
   
 





#source code of girvan_newman which is available in netowrkx. This code was modified  to make
#the parameter k in edge_betweenness_centrality equal to 10 to reduce the computation time                 
def girvan_newman_modified(G, most_valuable_edge=None):
    """Finds communities in a graph using the Girvan–Newman method.

    Parameters
    ----------
    G : NetworkX graph

    most_valuable_edge : function
        Function that takes a graph as input and outputs an edge. The
        edge returned by this function will be recomputed and removed at
        each iteration of the algorithm.

        If not specified, the edge with the highest
        :func:`networkx.edge_betweenness_centrality` will be used.

    Returns
    -------
    iterator
        Iterator over tuples of sets of nodes in `G`. Each set of node
        is a community, each tuple is a sequence of communities at a
        particular level of the algorithm.

    Examples
    --------
    To get the first pair of communities::

        >>> G = nx.path_graph(10)
        >>> comp = girvan_newman(G)
        >>> tuple(sorted(c) for c in next(comp))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

    To get only the first *k* tuples of communities, use
    :func:`itertools.islice`::

        >>> import itertools
        >>> G = nx.path_graph(8)
        >>> k = 2
        >>> comp = girvan_newman(G)
        >>> for communities in itertools.islice(comp, k):
        ...     print(tuple(sorted(c) for c in communities))  # doctest: +SKIP
        ...
        ([0, 1, 2, 3], [4, 5, 6, 7])
        ([0, 1], [2, 3], [4, 5, 6, 7])

    To stop getting tuples of communities once the number of communities
    is greater than *k*, use :func:`itertools.takewhile`::

        >>> import itertools
        >>> G = nx.path_graph(8)
        >>> k = 4
        >>> comp = girvan_newman(G)
        >>> limited = itertools.takewhile(lambda c: len(c) <= k, comp)
        >>> for communities in limited:
        ...     print(tuple(sorted(c) for c in communities))  # doctest: +SKIP
        ...
        ([0, 1, 2, 3], [4, 5, 6, 7])
        ([0, 1], [2, 3], [4, 5, 6, 7])
        ([0, 1], [2, 3], [4, 5], [6, 7])

    To just choose an edge to remove based on the weight::

        >>> from operator import itemgetter
        >>> G = nx.path_graph(10)
        >>> edges = G.edges()
        >>> nx.set_edge_attributes(G, {(u, v): v for u, v in edges}, "weight")
        >>> def heaviest(G):
        ...     u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
        ...     return (u, v)
        ...
        >>> comp = girvan_newman(G, most_valuable_edge=heaviest)
        >>> tuple(sorted(c) for c in next(comp))
        ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9])

    To utilize edge weights when choosing an edge with, for example, the
    highest betweenness centrality::

        >>> from networkx import edge_betweenness_centrality as betweenness
        >>> def most_central_edge(G):
        ...     centrality = betweenness(G, weight="weight")
        ...     return max(centrality, key=centrality.get)
        ...
        >>> G = nx.path_graph(10)
        >>> comp = girvan_newman(G, most_valuable_edge=most_central_edge)
        >>> tuple(sorted(c) for c in next(comp))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

    To specify a different ranking algorithm for edges, use the
    `most_valuable_edge` keyword argument::

        >>> from networkx import edge_betweenness_centrality
        >>> from random import random
        >>> def most_central_edge(G):
        ...     centrality = edge_betweenness_centrality(G)
        ...     max_cent = max(centrality.values())
        ...     # Scale the centrality values so they are between 0 and 1,
        ...     # and add some random noise.
        ...     centrality = {e: c / max_cent for e, c in centrality.items()}
        ...     # Add some random noise.
        ...     centrality = {e: c + random() for e, c in centrality.items()}
        ...     return max(centrality, key=centrality.get)
        ...
        >>> G = nx.path_graph(10)
        >>> comp = girvan_newman(G, most_valuable_edge=most_central_edge)

    Notes
    -----
    The Girvan–Newman algorithm detects communities by progressively
    removing edges from the original graph. The algorithm removes the
    "most valuable" edge, traditionally the edge with the highest
    betweenness centrality, at each step. As the graph breaks down into
    pieces, the tightly knit community structure is exposed and the
    result can be depicted as a dendrogram.

    """
    # If the graph is already empty, simply return its connected
    # components.
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:

        def most_valuable_edge(G):
            """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
            # We have guaranteed that the graph is non-empty, so this
            # dictionary will never be empty.
            betweenness = nx.edge_betweenness_centrality(G,k=10)
            return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy().to_undirected()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)



def _without_most_central_edges(G, most_valuable_edge):
    """Returns the connected components of the graph that results from
    repeatedly removing the most "valuable" edge in the graph.

    `G` must be a non-empty graph. This function modifies the graph `G`
    in-place; that is, it removes edges on the graph `G`.

    `most_valuable_edge` is a function that takes the graph `G` as input
    (or a subgraph with one or more edges of `G` removed) and returns an
    edge. That edge will be removed and this process will be repeated
    until the number of connected components in the graph increases.

    """
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components
     
        
     
        
    
 #excute the code        

###############################################################################


    
dblp2005, dblp2005w, dblp2006 = create_graphs()       

#calculate_PageRank(dblp2005,dblp2006,dblp2005w)   
    
#calculate_Betweenness(dblp2005,dblp2006,dblp2005w)
    
#dblp2005_core , dblp2006_core = create_core_graphs(dblp2005,dblp2006)

#fof(dblp2005_core)

#target_edges(dblp2005_core,dblp2006_core)
 
#random_predictor()  

#common_neighbors(dblp2005_core)

#jaccard_coefficient(dblp2005_core)

#preferential_attachment(dblp2005_core)

#adamic_adar(dblp2005_core)


#precision("random_predictor.txt", 10, "random_predictor_p@10.txt")
#precision("random_predictor.txt", 20, "random_predictor_p@20.txt")
#precision("random_predictor.txt", 50, "random_predictor_p@50.txt")
#precision("random_predictor.txt", 100, "random_predictor_p@100.txt")
#precision("random_predictor.txt", count_target_edges(), "random_predictor_p@T.txt")


#precision("graph_common_neighbors.txt", 10, "graph_common_neighbors_p@10.txt")
#precision("graph_common_neighbors.txt", 20, "graph_common_neighbors_p@20.txt")
#precision("graph_common_neighbors.txt", 50, "graph_common_neighbors_p@50.txt")
#precision("graph_common_neighbors.txt", 100, "graph_common_neighbors_p@100.txt")
#precision("graph_common_neighbors.txt", count_target_edges(), "graph_common_neighbors_p@T.txt")


#precision("graph_jaccard.txt", 10, "graph_jaccard_p@10.txt")
#precision("graph_jaccard.txt", 20, "graph_jaccard_p@20.txt")
#precision("graph_jaccard.txt", 50, "graph_jaccard_p@50.txt")
#precision("graph_jaccard.txt", 100, "graph_jaccard_p@100.txt")
#precision("graph_jaccard.txt", count_target_edges(), "graph_jaccard_p@T.txt")


#precision("graph_preferential_attachment.txt", 10, "graph_preferential_attachment_p@10.txt")
#precision("graph_preferential_attachment.txt", 20, "graph_preferential_attachment_p@20.txt")
#precision("graph_preferential_attachment.txt", 50, "graph_preferential_attachment_p@50.txt")
#precision("graph_preferential_attachment.txt", 100, "graph_preferential_attachment_p@100.txt")
#precision("graph_preferential_attachment.txt", count_target_edges(), "graph_preferential_attachment_p@T.txt")


#precision("graph_adamic_adar.txt", 10, "graph_adamic_adar_p@10.txt")
#precision("graph_adamic_adar.txt", 20, "graph_adamic_adar_p@20.txt")
#precision("graph_adamic_adar.txt", 50, "graph_adamic_adar_p@50.txt")
#precision("graph_adamic_adar.txt", 100, "graph_adamic_adar_p@100.txt")
#precision("graph_adamic_adar.txt", count_target_edges(), "graph_adamic_adar_p@T.txt")

#community_detection_in_graph(dblp2005)