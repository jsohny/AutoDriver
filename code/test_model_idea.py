"""
Attempt to make something that uses average data labels of
closest neighbor wrt. locations of all agents at start of 
intersection data plus an autoregressive model. 
"""

import utils
import numpy as np
from models.autoregression import TimeSeriesRegression
from collections import defaultdict

# First, load in training and validation examples
train_data_pairs = utils.load_dataset_with_labels("train")
# val_data_pairs = utils.load_dataset_with_labels("val")
test_data = utils.load_test_dataset()

# Now, determine the equivalence classes
data_classes = defaultdict(list)

for x, y in train_data_pairs:

    n_agents = utils.get_num_agents(x)
    n_cars = utils.get_num_cars(x)
    n_pedestrians = utils.get_num_pedestrians_bikes(x)

    starting_positions = utils.get_starting_positions(x)
    # print(starting_positions)
    data_classes[(n_agents, n_cars, n_pedestrians)].append((y.to_numpy()[:, 1:], starting_positions))

pred_actual_pairs = []
n_neighbors = 3
for x, file_name in test_data:

    n_agents = utils.get_num_agents(x)
    n_cars = utils.get_num_cars(x)
    n_pedestrians = utils.get_num_pedestrians_bikes(x)

    # Lookup the set of examples that are in the same class 
    equiv_class = data_classes[(n_agents, n_cars, n_pedestrians)]

    example_starting_positions = utils.get_starting_positions(x)
    neighbors = []
    for data_pair in equiv_class:
        if (data_pair[0].shape != (30, 2)):
            continue

        start_pos = data_pair[1]
        dist = np.linalg.norm(start_pos - example_starting_positions)
        if len(neighbors) < n_neighbors:
            neighbors.append((dist, data_pair[0]))
        else:
            dists = [x[0] for x in neighbors]
            for i in range(len(dists)):
                if dists[i] > dist:
                    neighbors[i] = (dist, data_pair[0])

    average_pred = np.zeros((30, 2))
    for dist, y_n in neighbors:
        average_pred += y_n
    average_pred /= len(neighbors)
        
    agent_idx = utils.get_ego_index(x)
    x_coords = x['x' + str(agent_idx)]
    y_coords = x['y' + str(agent_idx)]
    model_x = TimeSeriesRegression(lookback=3)
    model_y = TimeSeriesRegression(lookback=3)
    model_x.create(x_coords)
    model_y.create(y_coords)

    # We predict 30 steps into the future
    n_timesteps = 30
    x_preds = np.zeros((n_timesteps, 1))
    y_preds = np.zeros((n_timesteps, 1))

    for i in range(n_timesteps):
        x_preds[i] = model_x.predictnext()
        y_preds[i] = model_y.predictnext()

    final_preds = np.hstack((x_preds, y_preds))
    average_pred += final_preds
    average_pred /= 2

    n_cars = utils.get_num_cars(x)

    pred_actual_pairs.append((average_pred, file_name))


# error = 0
# for pred, actual, n_cars in pred_actual_pairs:
#     if (actual.shape != (30, 2)):
#         continue
#     error += np.sqrt((1/(2*n_cars*30))*np.sum((pred - actual)**2))

# print("Final validation error: %d" % error)

predictions = dict()
for pred, fname in pred_actual_pairs:
    test_output_index = fname[fname.index("_") + 1 : fname.index(".")]
    test_output_index = int(test_output_index)

    predictions[test_output_index] = pred

utils.write_predictions_to_file(predictions, "preds_test_idea.csv")
