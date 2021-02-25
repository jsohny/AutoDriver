"""This file is used to explore some possible models
We might want to try, and test things out"""

import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from models.autoregression import TimeSeriesRegression

# For visualizing the data
from sklearn.cluster import KMeans

# Use the load_dataset_with_labels function to grab a list of 
# (X, y) training data pairs, where X is a pandas dataframe containing the 
# features of one of the X_xxx.csv files and y is the corresponding
# label dataframe
# training_data_pairs = utils.load_dataset_with_labels("train")
# validation_data_pairs = utils.load_dataset_with_labels("val")
test_data = utils.load_test_dataset()

# print("Number of training examples: %d" % len(training_data_pairs))
# print("Number of validation examples: %d" % len(validation_data_pairs))
# print("Number of test data examples: %d" % len(test_data))

# X_0 = training_data_pairs[0][0]
# y_0 = training_data_pairs[0][1]

# agent = 1

# for i in range(10):
#     if agent in X_0["role" + str(i)].values:
#         agent_coords_x = X_0["x" + str(i)]
#         agent_coords_y = X_0["y" + str(i)]
#         break

# These are useful contextual features that we might want to use
# num_agents_present = utils.get_num_agents(X_0)
# print("There are %d agents in this example" % num_agents_present)
# num_cars_present = utils.get_num_cars(X_0)
# print("There are %d cars in this example" % num_cars_present)
# num_pedestrians_bikes = utils.get_num_pedestrians_bikes(X_0)
# print("There are %d pedestrians and bikes in this example" % num_pedestrians_bikes)

# closest_distance_to_ego_vehicle = utils.get_closest_distance_to_ego(X_0)
# print("Closest agent distance to ego is: %f" % closest_distance_to_ego_vehicle)

# Determine all the above 'derived' features and then 
# run KMeans clustering to visualize all the possible scenarios
# metadata = []
# for i in range(len(training_data_pairs)):
#     X = training_data_pairs[i][0]
#     n_agents = utils.get_num_agents(X)
#     num_cars_present = utils.get_num_cars(X)
#     num_pedestrians_bikes = utils.get_num_pedestrians_bikes(X)
#     # closest_dist = utils.get_closest_distance_to_ego(X)
#     metadata.append(np.array([n_agents, num_cars_present, num_pedestrians_bikes]))

# Use the elbow method to determine k
# squared_distances = []
# for k in range(1, 15):
#     kmeans = KMeans(n_clusters=k)
#     km = kmeans.fit(metadata)
#     squared_distances.append(km.inertia_)

# plt.plot(range(1, 15), squared_distances)
# plt.show()

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(metadata)
# clusters = kmeans.cluster_centers_

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# xs = [x[0] for x in cluster_centers]
# ys = [x[1] for x in cluster_centers]
# zs = [x[2] for x in cluster_centers]

# ax.scatter(xs, ys, zs)
# ax.set_xlabel('num_agents')
# ax.set_ylabel('num_cars_present')
# ax.set_zlabel('num_pedestrians_bikes')
# plt.show()

# Try something really basic - just train separate model for 
# x coordinates and y coordinates, then predict x coord, then y coord
# using autoregressive model

# model_x = TimeSeriesRegression(lookback=3)
# model_x.create(agent_coords_x)
# model_y = TimeSeriesRegression(lookback=3)
# model_y.create(agent_coords_y)

# # We predict 30 steps into the future
# n_timesteps = 30
# x_preds = np.zeros((n_timesteps, 1))
# y_preds = np.zeros((n_timesteps, 1))

# for i in range(n_timesteps):
#     x_preds[i] = model_x.predictnext()
#     y_preds[i] = model_y.predictnext()

# num_cars = utils.get_num_cars(X_0)
# final_preds = np.hstack((x_preds, y_preds))
# # # # print(final_preds)
# e_train = np.sum((1/(2*num_cars*n_timesteps)) * np.sqrt((final_preds - y_0.to_numpy()[:, 1:])**2))
# print("Training error for single example: %3f" % e_train)

# Now, we can determine all the examples that correspond to each 'cluster'
# and train autoregressive models for each. At prediction time, 
# just determine the cluster that an example belongs to and
# use the corresponding autoregressive model to make the predictions.
# Conceptually, this should make sense, since examples in a similar 
# cluster should (hopefully) have similar types of situations occur

# Alternatively, and maybe better, is a decision tree approach, since it 
# will probably provide more fine grained control over which model should
# be used in a given scenario.

# predictions = dict()
# for idx, tup in enumerate(test_data):
#     df_X = tup[0]
#     example_file_name = tup[1]
#     test_output_index = example_file_name[example_file_name.index("_") + 1 : example_file_name.index(".")]
#     test_output_index = int(test_output_index)

#     agent_idx = utils.get_ego_index(df_X)
#     agent_coords_x = df_X['x' + str(agent_idx)]
#     agent_coords_y = df_X['y' + str(agent_idx)]

#     model_x = TimeSeriesRegression(lookback=3)
#     model_x.create(agent_coords_x)
#     model_y = TimeSeriesRegression(lookback=3)
#     model_y.create(agent_coords_y)

#     # We predict 30 steps into the future
#     n_timesteps = 30
#     x_preds = np.zeros((n_timesteps, 1))
#     y_preds = np.zeros((n_timesteps, 1))

#     for i in range(n_timesteps):
#         x_preds[i] = model_x.predictnext()
#         y_preds[i] = model_y.predictnext()

#     num_cars = utils.get_num_cars(df_X)
#     final_preds = np.hstack((x_preds, y_preds))

#     predictions[test_output_index] = final_preds
    
# utils.write_predictions_to_file(predictions, "preds.csv")


