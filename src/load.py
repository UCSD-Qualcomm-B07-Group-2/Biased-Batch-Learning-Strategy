import torch
import gzip
import json
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix

def load_data(i):
    if os.path.exists('../cache/data.pt'):
        # Load data from cache
        data = torch.load('../cache/data.pt')
        return data
    else:
        with gzip.open(f'../data/xbar/{i}/xbar.json.gz','rb') as f:
            design = json.loads(f.read().decode('utf-8'))
            
        instances = pd.DataFrame(design['instances'])
        nets = pd.DataFrame(design['nets'])

        def buildBST(array,start=0,finish=-1):
            if finish<0:
                finish = len(array)
            mid = (start + finish) // 2
            if mid-start==1:
                ltl=start
            else:
                ltl=buildBST(array,start,mid)
            
            if finish-mid==1:
                gtl=mid
            else:
                gtl=buildBST(array,mid,finish)
                
            return((array[mid],ltl,gtl))

        congestion_data = np.load('../data/xbar/1/xbar_congestion.npz')
        xbst=buildBST(congestion_data['xBoundaryList'])
        ybst=buildBST(congestion_data['yBoundaryList'])
        demand = np.zeros(shape = [instances.shape[0],])

        def getGRCIndex(x,y,xbst,ybst):
            while (type(xbst)==tuple):
                if x < xbst[0]:
                    xbst=xbst[1]
                else:
                    xbst=xbst[2]
                    
            while (type(ybst)==tuple):
                if y < ybst[0]:
                    ybst=ybst[1]
                else:
                    ybst=ybst[2]
                    
            return ybst, xbst


        for k in range(instances.shape[0]):
            xloc = instances.iloc[k]['xloc']; yloc = instances.iloc[k]['yloc']
            i,j=getGRCIndex(xloc,yloc,xbst,ybst)
            d = 0 
            for l in list(congestion_data['layerList']): 
                lyr=list(congestion_data['layerList']).index(l)
                d += congestion_data['demand'][lyr][i][j]
            demand[k] = d
                
        instances['routing_demand'] = demand

        conn = np.load(f'../data/xbar/1/xbar_connectivity.npz')
        A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
        A = A.__mul__(A.T)

        X = torch.tensor(instances[['xloc', 'yloc', 'cell', 'orient']].values) # 4 features
        y = torch.tensor(instances['routing_demand'].values) # y value
        edges = from_scipy_sparse_matrix(A)
        edge_index = edges[0]
        edge_weights = edges[1]

        # get all unique nodes in edges
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_weights, y=y)
        os.mkdir('../cache')
        torch.save(data, '../cache/data.pt')
        return data