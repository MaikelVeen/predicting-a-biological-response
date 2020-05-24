import pandas as pd

train_data = pd.read_csv("train.csv", header=[0])
test_data = pd.read_csv("test.csv", header=[0])

response = train_data.loc[train_data["Activity"] == 1].drop("Activity", axis = 1)
no_response = train_data.loc[train_data["Activity"] == 0].drop("Activity", axis = 1)

X = no_response.mean()
Y = response.mean()

max_diff = 0
max_index = 0
diff_array = []

for i in range(len(X)):
    diff = (X[i] - Y[i])**2
    diff_array.append(diff)
    if diff > max_diff:
        max_index = i
        max_diff = diff

diff_array = pd.DataFrame({"squared difference" : diff_array})
n_smallest_diff = diff_array.nsmallest(50, "squared difference").index
remove_treshold = []

for index in n_smallest_diff:
    remove_treshold.append("D" + str(index + 1))

print(remove_treshold)