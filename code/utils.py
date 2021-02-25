import pandas as pd
import numpy as np
import os
import math

dataset_path = "../dataset/"
# Loads a dataset, given the name of the dataset to load
# name is either 'train', 'val', or 'test'
# If name is train or val, X and y are the matrices representing
# features and desired outputs - otherwise y is omitted 
def load_dataset(name):
    if name == "train" or name == "val":
        X = load_dataset_with_labels(name)
    elif name == "test":
        X = load_test_dataset()
    else:
        raise Exception("Invalid dataset name provided")

    return X

def load_dataset_with_labels(name):
    starting_dir = os.getcwd()
    os.chdir(dataset_path + name)
    os.chdir("X")
    X_listing = os.listdir()
    os.chdir("../")
    os.chdir("y")
    y_listing = os.listdir()
    os.chdir("../")

    dataframe_pairs = []
    for x, y in zip(X_listing, y_listing):
        df_X = pd.read_csv("X/" + x)
        df_X = sanitize_dataframe(df_X)

        df_y = pd.read_csv("y/" + y)
        dataframe_pairs.append((df_X, df_y))

    os.chdir(starting_dir)
    
    return dataframe_pairs

# Strips the whitespace from column names
# strips the whitespace from column values
# then replaces each categorical variable (role/type)
# with a numerical value
def sanitize_dataframe(df_X):
    df_X.rename(columns=lambda x: x.strip(), inplace=True)
    df_X = df_X.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df_X.replace(['agent', 'others'], [1, 0], inplace=True)

    # TODO: Are there other values possible here? It's hard to tell
    # from just combing through the .csv files, and these are the 
    # only ones mentioned in the kaggle description
    df_X.replace(['car', 'pedestrian/bicycle', 0.0, '0.0'], [2, 1, 0, 0], inplace=True)

    # NOTE: Since I'm just returning the dataframe now I commented this out
    # Remove the 'P' that appears before the 
    # ids of pedestrians.
    cols = [c for c in df_X.columns if c[:2] != 'id']
    df_X = df_X[cols]

    # columns = list(df_X.columns.values)
    # for column_name in columns:
    #     if 'id' in column_name:
    #         df_X[column_name] = df_X[column_name].map(lambda x: str(x).lstrip('P'))

    return df_X

def load_test_dataset():
    starting_dir = os.getcwd()
    os.chdir(dataset_path + "val")
    os.chdir("X")
    listing = os.listdir()

    dataframes = []
    for x in listing:
        df_X = pd.read_csv(x)
        
        df_X = sanitize_dataframe(df_X)
        
        dataframes.append((df_X, x))

    os.chdir(starting_dir)

    return dataframes

def load_test_labels():
    starting_dir = os.getcwd()
    os.chdir(dataset_path + "val")
    os.chdir("Y")
    listing = os.listdir()
    
    dataframes = []
    for x in listing:
        df_X = pd.read_csv(x)
        
        dataframes.append((df_X, x))

    os.chdir(starting_dir)

    return dataframes


def get_num_agents(df_X):
    n_agents = 0

    col_names = list(df_X.columns.values)
    for name in col_names:
        if "present" in name:
            if not (df_X[name] == 0).all():
                n_agents += 1
    
    # Return number found minus one since 
    # the ego vehicle will always be present
    return n_agents - 1

def get_num_cars(df_X):
    n_cars = 0

    col_names = list(df_X.columns.values)
    for name in col_names:
        if "type" in name:
            if (df_X[name] == 2).any():
                n_cars += 1
    
    # Return number found minus one since 
    # the ego vehicle will always be present
    return n_cars - 1

def get_num_pedestrians_bikes(df_X):
    n_pedestrians_bikes = 0

    col_names = list(df_X.columns.values)
    for name in col_names:
        if "type" in name:
            if (df_X[name] == 1).any():
                n_pedestrians_bikes += 1
    
    return n_pedestrians_bikes

# Returns the index of the ego vehicle
# in the pandas dataframe
def get_ego_index(df_X):
    col_names = [x for x in df_X.columns.values if "role" in x]
    for idx, name in enumerate(col_names):
        if (df_X[name] == 1).any():
            return idx

    raise Exception("Ego index was not found in the dataframe")
                
# Returns the closest starting distance
# to the ego vehicle's starting position.
# This might be helpful 
def get_closest_distance_to_ego(df_X):
    # First, find the ego's (x, y)
    ego_idx = get_ego_index(df_X)
    ego_X = df_X["x" + str(ego_idx)].to_numpy()[0]
    ego_Y = df_X["y" + str(ego_idx)].to_numpy()[0]

    best_distance = None
    for i in range(10):
        if i != ego_idx:
            this_X = df_X["x" + str(i)].to_numpy()[0]
            this_Y = df_X["y" + str(i)].to_numpy()[0]
            dist = calculate_distance(ego_X, ego_Y, this_X, this_Y)

            if best_distance is None or best_distance > dist:
                best_distance = dist
    
    return dist

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Given a dataframe, return all 
# the starting positions of each agent
def get_starting_positions(df_X):
    starting_positions = []
    cols_x = [x for x in df_X.columns.values if x[0] == "x"]
    cols_y = [y for y in df_X.columns.values if y[0] == "y"]
    for x, y in zip(cols_x, cols_y):
        starting_positions.append(df_X[x][0])
        starting_positions.append(df_X[y][0])
    
    return np.array(starting_positions)


# Write a dictionary containing the prediction [x,y]
# hstack'ed matrix to an output file for submission to Kaggle
def write_predictions_to_file(predictions, fname):
    with open("../predictions/" + fname, 'w') as f:
        f.write('id,location\n')
        for idx, elem in predictions.items():
            i = 1
            for x, y in zip(elem[:, 0], elem[:, 1]):
                f.write(str(idx) + '_x_' + str(i) + ',' + str(x) + '\n')
                f.write(str(idx) + '_y_' + str(i) + ',' + str(y) + '\n')
                i += 1