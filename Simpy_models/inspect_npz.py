# import numpy as np

# z = np.load("data/lstm_windows.npz", allow_pickle=True)
# for split in ["y_train","y_val","y_test"]:
#     y = z[split]
#     print(split, "dtype=", y.dtype, "unique=", sorted(set(y.tolist()))[:20])
# for key in z.files:
#     if "label" in key.lower() or "class" in key.lower():
#         print("Found key:", key, "->", z[key])





# Fast inspection: keys + shapes
# import numpy as np
# npz = np.load("data/lstm_windows.npz", allow_pickle=True)
# print("Keys:", npz.files)
# for k in npz.files:
#     arr = npz[k]
#     if hasattr(arr, "shape"):
#         print(k, arr.shape, arr.dtype)
#     else:
#         print(k, type(arr))



# printing label meanings and counts
# import numpy as np
# from collections import Counter

# npz = np.load("data/lstm_windows.npz", allow_pickle=True)
# label_map = list(npz["label_map"])
# y_train = npz["y_train"]

# print("Label map:", label_map)
# print("Train label counts:", {label_map[i]: c for i,c in Counter(y_train).items()})



# Inspecting feature schema 
# import numpy as np
# npz = np.load("data/lstm_windows.npz", allow_pickle=True)
# feature_names = list(npz["feature_names"])

# print("Number of features:", len(feature_names))
# print("First 25 features:")
# for f in feature_names[:25]:
#     print(" -", f)


# Looking at one window sample
# npz = np.load("data/lstm_windows.npz", allow_pickle=True)
# X = npz["X_train"]
# y = npz["y_train"]
# rid = npz["rid_train"]
# t0 = npz["t0_train"]
# i = 0
# print("Window shape:", X[i].shape)   # (60, F)
# print("Label:", y[i])
# print("Run:", rid[i])
# print("Start time:", t0[i])
# print("First timestep (first 10 features):", X[i,0,:10])


# Convert one window to a pandas DataFrame (for plotting)
# import pandas as pd
# import numpy as np
# npz = np.load("data/lstm_windows.npz", allow_pickle=True)
# X = npz["X_train"]
# feature_names = list(npz["feature_names"])
# df = pd.DataFrame(X[0], columns=feature_names)
# print(df.head())



