from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle as pk
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import *


x, y = load_data()  # load the dataset
data = split_data(x, y, test_size=0.1, valid_size=0.1)  # split the data into training, validation and testing sets


sc = MinMaxScaler()
data["x_train"] = sc.fit_transform(data["x_train"])
data["x_test"] = sc.transform(data["x_test"])
data["x_valid"] = sc.transform(data["x_valid"])
pca = PCA(n_components=0.95)
data["x_train"] = pca.fit_transform(data["x_train"])
data["x_test"] = pca.transform(data["x_test"])
data["x_valid"] = pca.transform(data["x_valid"])

if not os.path.isdir("tools"):
    os.mkdir("tools")
pk.dump(pca, open("tools/PCA.pkl", "wb"))
pk.dump(sc, open("tools/MinMaxScaler.pkl", "wb"))



age_weights = compute_weight(data["y_train"])

model = create_model(data["x_train"].shape[1])
tensorboard = TensorBoard(log_dir="logs") # use tensorboard to view metrics
early_stopping = EarlyStopping(mode="min", patience=10, restore_best_weights=True) # early stop after 10 non-improving epochs

batch_size = 400
epochs = 120

# train the model using the training set and validating using validation set
model.fit(data["x_train"], data["y_train"], epochs=epochs, batch_size=batch_size, class_weight=age_weights,
          validation_data=(data["x_valid"], data["y_valid"]), callbacks=[tensorboard, early_stopping])
model.save("results/model.h5") # save the model to a file

# evaluating the model using the testing set
print(f"Evaluating the model using {len(data['x_test'])} samples...")
loss, accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

