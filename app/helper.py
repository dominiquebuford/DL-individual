import numpy as np 
import pandas as pd
from scipy.spatial.distance import euclidean

def normalize(row, lenBody_x, lenBody_y):
    row_df = pd.DataFrame(row).transpose()
    excluded_columns = ['center_X','center_Y']

    for column in row_df.columns:
        if column not in excluded_columns:
            if (column.find('_X') != -1) and ((row_df[column]!=-1).all()):
                row_df[column] = (row_df[column] - row_df['center_X'])/lenBody_x
            if (column.find('_Y') != -1) and ((row_df[column]!=-1).all()):
                row_df[column] = (row_df[column] - row_df['center_Y'])/lenBody_y
    return row_df.squeeze()

def calculate_distance(point1, point2):
    if -1 in point1 or -1 in point2:
        return -1
    return euclidean(point1, point2)

def point_to_point_distance(df):
    df_new = pd.DataFrame()
    df_new['r ankle'] = df.apply(lambda row: (row['r ankle_X'], row['r ankle_Y']), axis=1)
    df_new['r knee'] = df.apply(lambda row: (row['r knee_X'], row['r knee_Y']), axis=1)
    df_new['r hip'] = df.apply(lambda row: (row['r hip_X'], row['r hip_Y']), axis=1)
    df_new['l hip'] = df.apply(lambda row: (row['l hip_X'], row['l hip_Y']), axis=1)
    df_new['l knee'] = df.apply(lambda row: (row['l knee_X'], row['l knee_Y']), axis=1)
    df_new['l ankle'] = df.apply(lambda row: (row['l ankle_X'], row['l ankle_Y']), axis=1)
    df_new['pelvis'] = df.apply(lambda row: (row['pelvis_X'], row['pelvis_Y']), axis=1)
    df_new['thorax'] = df.apply(lambda row: (row['thorax_X'], row['thorax_Y']), axis=1)
    df_new['upper neck'] = df.apply(lambda row: (row['upper neck_X'], row['upper neck_Y']), axis=1)
    df_new['head top'] = df.apply(lambda row: (row['head top_X'], row['head top_Y']), axis=1)
    df_new['r wrist'] = df.apply(lambda row: (row['r wrist_X'], row['r wrist_Y']), axis=1)
    df_new['r elbow'] = df.apply(lambda row: (row['r elbow_X'], row['l elbow_Y']), axis=1)
    df_new['r shoulder'] = df.apply(lambda row: (row['r shoulder_X'], row['r shoulder_Y']), axis=1)
    df_new['l shoulder'] = df.apply(lambda row: (row['l shoulder_X'], row['l shoulder_Y']), axis=1)
    df_new['l elbow'] = df.apply(lambda row: (row['l elbow_X'], row['l elbow_Y']), axis=1)
    df_new['l wrist'] = df.apply(lambda row: (row['l wrist_X'], row['l wrist_Y']), axis=1)

    distances_list = []

    # Iterate over each row
    for index, row in df_new.iterrows():
        check = []
        # Dictionary to store distances for the current row
        row_distances = {}
        # Iterate over each column pair
        for column1 in df_new.columns:
            for column2 in df_new.columns:
                tupletoAdd = (column2, column1)
                check.append(tupletoAdd)
                if column1 != column2 and (column1, column2) not in check:
                    # Calculate distance between current pair of columns
                    distance = calculate_distance(row[column1], row[column2])
                    # Construct the name for the new distance column
                    distance_column_name = f'{column1}_{column2}_distance'
                    # Store the distance in the dictionary
                    row_distances[distance_column_name] = distance
        # Append the distances for the current row to the list
        distances_list.append(row_distances)

    # Create a DataFrame from the list of dictionaries
    distances_df = pd.DataFrame(distances_list)
    return distances_df

def preprocess_keypoints(keypoints):
    x_values = [keypoint[0] for keypoint in keypoints]
    y_values = [keypoint[1] for keypoint in keypoints]

    lenBody_x = max(x_values) - min(x_values)
    lenBody_y = max(y_values) - min(y_values)
    labels= ['r ankle','r knee', 'r hip', 'l hip', 'l knee', 'l ankle', 'pelvis', 'thorax', 'upper neck', 'head top', 'r wrist', 'r elbow', 'r shoulder', 'l shoulder', 'l elbow', 'l wrist']
    new_labels = []
    for label in labels:
        new_labels.extend([f"{label}_X", f"{label}_Y"])
    new_keypoints = []
    for keypoint in keypoints:
        if keypoint[2]<0.05:    #might change threshold
            new_keypoints.extend([-1, -1])
        else:
            new_keypoints.extend([keypoint[0], keypoint[1]])
    

    df = pd.DataFrame([new_keypoints], columns = new_labels)
    #find center
    df['center_X'] = (df['r hip_X'] + df['l hip_X'])/2
    df['center_Y'] = (df['r hip_Y'] + df['l hip_Y']) / 2

    #normalize the values
    df_normalized = df.apply(lambda row: normalize(row, lenBody_x, lenBody_y), axis=1)
    df_final = df_normalized.drop(columns = ['center_X', 'center_Y'])

    #df_to_return = point_to_point_distance(df_final)
    return df_final.to_numpy().astype(np.float32)



