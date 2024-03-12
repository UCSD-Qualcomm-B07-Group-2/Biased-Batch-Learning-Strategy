import gzip
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from torch_geometric.data import Batch

def inductive_split(data_list, train_prop, val_prop, test_prop, features):
    # Calculate the number of samples for each split
    total_samples = len(data_list)
    train_samples = int(total_samples * train_prop)
    val_samples = int(total_samples * val_prop)
    test_samples = int(total_samples * test_prop)

    # Split the data list into train, val, and test sets
    train_set = data_list[:train_samples]
    val_set = data_list[train_samples:train_samples+val_samples]
    test_set = data_list[train_samples+val_samples:]

    # Select the desired features from the data sets
    train_set = [Data(x=d.x[features], edge_index=d.edge_index, edge_attr=d.edge_weights, y=d.y) for d in train_set]
    val_set = [Data(x=d.x[features], edge_index=d.edge_index, edge_attr=d.edge_weights, y=d.y) for d in val_set]
    test_set = [Data(x=d.x[features], edge_index=d.edge_index, edge_attr=d.edge_weights, y=d.y) for d in test_set]

    # Convert the data sets to DataLoader objects with batch size 1
    train_loader = DataLoader(train_set, batch_size=1)
    val_loader = DataLoader(val_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)

    return train_loader, val_loader, test_loader



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

def get_xbar_data(i):
    with gzip.open(f'../data/xbar/{i}/xbar.json.gz','rb') as f:
        design = json.loads(f.read().decode('utf-8'))

    instances = pd.DataFrame(design['instances'])
    nets = pd.DataFrame(design['nets'])

    conn=np.load(f'../data/xbar/{i}/xbar_connectivity.npz')
    A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
    A = A.__mul__(A.T)
    A.setdiag(0)
    A = (A >= 1).astype(int)

    congestion_data = np.load(f'../data/xbar/{i}/xbar_congestion.npz')
    xbst = buildBST(congestion_data['xBoundaryList'])
    ybst = buildBST(congestion_data['yBoundaryList'])
    demand = np.zeros(shape = [instances.shape[0]])

    for k in range(instances.shape[0]):
        # print(k)
        xloc = instances.iloc[k]['xloc']; yloc = instances.iloc[k]['yloc']
        i,j=getGRCIndex(xloc,yloc,xbst,ybst)
        d = 0 
        for l in list(congestion_data['layerList']): 
            lyr=list(congestion_data['layerList']).index(l)
            d += congestion_data['demand'][lyr][i][j]
        demand[k] = d

    instances['routing_demand'] = demand
    return A, instances

def get_cells(circuit):
    #takes in 'superblue' or 'xbar'
    with gzip.open(f'../data/{circuit}/cells.json.gz', 'rb') as f:
        cells_json = json.loads(f.read())
    cells = pd.DataFrame(cells_json).drop(columns=['name'])
    return cells

# cells = get_cells('xbar')
# cell_ohe = OneHotEncoder(drop='first')
# cell_ohe.fit(cells[['id']])

def create_data_object(A, df):
    
    transformed = cell_ohe.transform(df[['cell']].rename(columns={'cell':'id'}))
    ohe_df = pd.DataFrame.sparse.from_spmatrix(transformed)
    df = df.merge(ohe_df, left_index=True, right_index=True)
    
    orient_ohe = OneHotEncoder(drop='first')
    transformed = orient_ohe.fit_transform(df[['orient']])
    ohe_df = pd.DataFrame.sparse.from_spmatrix(transformed)
    df = df.merge(ohe_df, left_index=True, right_index=True)
    
    GRC_widths = df.groupby('cell')['xloc'].max() - df.groupby('cell')['xloc'].min()
    GRC_heights = df.groupby('cell')['yloc'].max() - df.groupby('cell')['yloc'].min()
    GRC_area = GRC_widths * GRC_heights
    GRC_area = GRC_area.replace(0, np.nan)  # Replace 0 areas with NaN
    terminal_density = df.groupby('cell')['id'].count() / GRC_area
    terminal_density = terminal_density.fillna(0)
    terminal_df = pd.DataFrame(terminal_density.rename('pin_density')).reset_index()
    df = df.merge(terminal_df, how='left', on='cell')

    # normalizing xloc yloc with min-max scaling
    df['normalized_x'] = (df['xloc'] - df['xloc'].max()) / (df['xloc'].max() - df['xloc'].min())
    df['normalized_y'] = (df['yloc'] - df['yloc'].max()) / (df['yloc'].max() - df['yloc'].min())

    # remove self loops
    A.setdiag(0)
    # binarize adjacency matrix
    A = (A >= 1).astype(int)
    # get degree
    df['degree'] = np.array(A.sum(axis=1)).flatten()
    df = df.drop(columns=['name', 'id', 'cell', 'orient', 'xloc', 'yloc'])
        
    X = torch.tensor(df.drop(columns=['routing_demand']).values, dtype=torch.float)
    y = torch.tensor(df['routing_demand'].values, dtype=torch.float)

    ei = from_scipy_sparse_matrix(A)
    edge_index = ei[0].to(dtype=torch.long) 
    data = Data(x=X, edge_index=edge_index, y=y)
    return data

def merge_datasets(datasets):
    # This function will merge a list of datasets into a single dataset
    merged_data_list = []
    for data in datasets:
        merged_data_list.append(data)
    return Batch.from_data_list(merged_data_list)

def test(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(predictions.squeeze(), data.y.float())
    return loss.item()  # Return the loss value