import utils
import pandas as pd
import numpy as np
from models.autoregression import TimeSeriesRegression

# Use the load_dataset_with_labels function to grab a list of 
# (X, y) training data pairs, where X is a pandas dataframe containing the 
# features of one of the X_xxx.csv files and y is the corresponding
# label dataframe

# training_data_pairs = utils.load_dataset_with_labels("train")
# validation_data_pairs = utils.load_dataset_with_labels("val")
test_data = utils.load_test_dataset()
test_labels = utils.load_test_labels()

# print(training_data_pairs[0][0].shape)
# print("Number of training examples: %d" % len(training_data_pairs))
# print("Number of validation examples: %d" % len(validation_data_pairs))
# print("Number of test data examples: %d" % len(test_data))


# Naive model as baseline - train individual 
# autoregressive models on x and y coordinates, 
# then use these to create predictions and save to file.
predictions = dict()

for idx, tup in enumerate(test_data):
    
    df_X = tup[0]
    #tup[0] contains the data 
    #tup[1] is the file name ex. X_2.csv
    
    
    example_file_name = tup[1]
    test_output_index = example_file_name[example_file_name.index("_") + 1 : example_file_name.index(".")]
    test_output_index = int(test_output_index)
    #test_output_index is the file number ex. 2 for X_2.csv

    agent_idx = utils.get_ego_index(df_X)
    #agent_idx is which index in file corresponds to ego data

    agent_coords_x = df_X['x' + str(agent_idx)]
    agent_coords_y = df_X['y' + str(agent_idx)]
    #agent_coords_x and y contains the dataframe which holds 
    #coordinates at certain time instance as well as the agent 
    # index (ex. Name: x8)

    #create timeSeriesRegression models for x and y coordinates separately
    model_x = TimeSeriesRegression(lookback=3)
    model_x.create(agent_coords_x)
    model_y = TimeSeriesRegression(lookback=3)
    model_y.create(agent_coords_y)

    # We predict 30 steps into the future
    n_timesteps = 30
    x_preds = np.zeros((n_timesteps, 1))
    y_preds = np.zeros((n_timesteps, 1))

    for i in range(n_timesteps):
        x_preds[i] = model_x.predictnext()
        y_preds[i] = model_y.predictnext()

    num_cars = utils.get_num_cars(df_X)
    final_preds = np.hstack((x_preds, y_preds))

    # print("errors?: ", x_preds)

    predictions[test_output_index] = final_preds
    
utils.write_predictions_to_file(predictions, "output")

for idx, tup in enumerate(test_labels):


    future_values = tup[0].values
    x_coords = np.zeros((n_timesteps, 1))
    y_coords = np.zeros((n_timesteps, 1))

    for step in range(n_timesteps):
        x_coords[step] = future_values[step][1]
        y_coords[step] = future_values[step][2]
    print(x_coords, y_coords)

