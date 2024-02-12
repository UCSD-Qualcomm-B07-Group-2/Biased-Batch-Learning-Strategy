import torch
import gzip
import json
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from tqdm import tqdm

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

def load_data(dataset):
    data_list = []
    for m in tqdm(range(1,4)):  # Assuming there are 5 files
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', f'data_{m}.pt')):
            # Load data from cache
            data = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', f'data_{m}.pt'))
            data_list.append(data)
        else:

            print(m)
            with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'xbar', str(m), 'xbar.json.gz')) as f:
                design = json.loads(f.read().decode('utf-8'))

                instances = pd.DataFrame(design['instances'])
                nets = pd.DataFrame(design['nets'])

            print('Loaded instances and nets')


            congestion_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'xbar', str(m), 'xbar_congestion.npz'))
            print('Loaded congestion data')
            xbst=buildBST(congestion_data['xBoundaryList'])
            ybst=buildBST(congestion_data['yBoundaryList'])
            demand = np.zeros(shape = [instances.shape[0],])

            for k in range(instances.shape[0]):
                xloc = instances.iloc[k]['xloc']; yloc = instances.iloc[k]['yloc']
                i,j=getGRCIndex(xloc,yloc,xbst,ybst)
                d = 0 
                for l in list(congestion_data['layerList']): 
                    lyr=list(congestion_data['layerList']).index(l)
                    d += congestion_data['demand'][lyr][i][j]
                demand[k] = d

            instances['routing_demand'] = demand

            print('Finished routing demand calculation')

            conn = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'xbar', str(m), 'xbar_connectivity.npz'))
            A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
            A = A.__mul__(A.T)

            X = torch.tensor(instances[['xloc', 'yloc', 'cell', 'orient']].values) # 4 features
            y = torch.tensor(instances['routing_demand'].values) # y value
            edges = from_scipy_sparse_matrix(A)
            edge_index = edges[0]
            edge_weights = edges[1]

            print('Loaded connectivity data')

            # get all unique nodes in edges
            data = Data(x=X, edge_index=edge_index, edge_attr=edge_weights, y=y)
            os.makedirs('../cache', exist_ok=True)
            torch.save(data, f'../cache/data_{m}.pt')
            data_list.append(data)
            print('Saved data')

    return data_list