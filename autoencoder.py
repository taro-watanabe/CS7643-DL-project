import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model


class UnivariateAutoencoder:
    def __init__(self, X, encoding_dim):
        self.encoding_dim = encoding_dim
        self.timesteps = X.shape[1]
        # Define the autoencoder model
        input_layer = Input(shape=(self.timesteps, 1))
        encoded = Dense(self.encoding_dim, activation="relu")(input_layer)
        decoded = Dense(1, activation="linear")(encoded)
        self.autoencoder = Model(input_layer, decoded)

        # Compile the autoencoder model
        self.autoencoder.compile(optimizer="adam", loss="mse")

    def fit(self, X):
        # standardize the data
        X = (X - np.mean(X)) / np.std(X)
        X = X.reshape(-1, self.timesteps, 1)
        # Fit the autoencoder model to the input data X
        self.autoencoder.fit(X, X, epochs=100, batch_size=64, shuffle=True)

    def predict(self, X, threshold):
        # Predict the reconstruction error for the input data X
        errors = X - self.autoencoder.predict(X)
        # Compute the mean squared error of the errors
        mse = np.mean(np.power(errors, 2), axis=1)
        # Identify anomalies by comparing the MSE to a threshold
        is_anomaly = mse > threshold
        return is_anomaly


class MultivariateAutoencoder:
    def __init__(self, X, num_features, encoding_dim):
        self.num_features = num_features
        self.encoding_dim = encoding_dim

        # Reshape input data to (num_time_steps, num_features)
        self.X_reshaped = X.reshape(-1, 1, num_features)
        print(self.X_reshaped.shape)

        # Define the autoencoder model
        input_layer = Input(shape=(None, num_features))
        encoded = LSTM(encoding_dim, activation="relu")(input_layer)
        decoded = RepeatVector(X.shape[1])(encoded)
        decoded = LSTM(num_features, activation="linear", return_sequences=True)(
            decoded
        )
        self.autoencoder = Model(input_layer, decoded)

        # Compile the autoencoder model
        self.autoencoder.compile(optimizer="adam", loss="mse")

    def fit(self):
        # Fit the autoencoder model to the input data X
        self.autoencoder.fit(self.X_reshaped, self.X_reshaped, epochs=10, batch_size=64, shuffle=True)

    def predict(self, X, threshold):
        # Reshape input data to (num_time_steps, num_features)
        X_reshaped = X.reshape(-1, 1, self.num_features)

        # Predict the reconstruction error for the input data X
        errors = X_reshaped - self.autoencoder.predict(X_reshaped)
        # Compute the mean squared error of the errors
        mse = np.mean(np.power(errors, 2), axis=(1, 2))
        # Identify anomalies by comparing the MSE to a threshold
        is_anomaly = mse > threshold
        return is_anomaly


def main(dir):
    def _predict_with_threshold(X, threshold_start):
        threshold = threshold_start
        is_anomaly = autoencoder.predict(X, threshold)
        print(len(np.where(is_anomaly)[0]))
        print(len(np.where(is_anomaly)[0]) / len(X) * 100, "%")
        c = 0
        while len(np.where(is_anomaly)[0]) / len(X) * 100 > 1:
            c += 1
            threshold *= 1 + c / 100  # sweet way to exponentially increase threshold
            is_anomaly = autoencoder.predict(X, threshold)
            print("Adjusting threshold to: ", threshold)
            print(len(np.where(is_anomaly)[0]))
            print(len(np.where(is_anomaly)[0]) / len(X) * 100, "%")
        return is_anomaly

    # def _predict_with_threshold(X, t):
    #     return autoencoder.predict(X, t)

    df = pd.read_csv(
        "datasets/csv_datasets/open-anomaly-detection-benchmark/datasets/" + dir
    )
    print(df.head())

    if "value" in df.columns:
        print("detected univariate dataset")
        X = df["value"].values.reshape(-1, 1)
        # Train the autoencoder model
        autoencoder = UnivariateAutoencoder(X, encoding_dim=5)
        autoencoder.fit(X)

        # Predict the reconstruction error for the input data X
        threshold_start = 0.1
        is_anomaly = _predict_with_threshold(X, threshold_start)
        print("Anomaly indices: ", np.where(is_anomaly))

        # load original data as df
        df = pd.read_csv(
            "datasets/csv_datasets/open-anomaly-detection-benchmark/datasets/" + dir
        )
        # add anomaly column to df
        df["anomaly"] = is_anomaly

        # save df to csv
        df.to_csv(
            "results_junkbox/autoencoder_results/" + dir + "_autoencoder.csv",
            index=False,
        )

        # save plot of anomaly
        fig, ax = plt.subplots(figsize=(30, 6))
        ax.plot(df["timestamp"], df["value"], color="gray")
        # actual anomalies vertical lines
        for idx in np.where(df["is_anomaly"])[0]:
            print(idx)
            print("found!")
            ax.axvline(df["timestamp"][idx], color="red", alpha=0.5)
        # predicted anomalies vertical lines
        for idx in np.where(df["anomaly"])[0]:
            ax.axvline(df["timestamp"][idx], color="blue", alpha=0.5)
        plt.title(dir + " autoencoder anomaly detection")

        # save plot
        plt.savefig("results_junkbox/autoencoder_results/" + dir + "_autoencoder.png")

    else:
        print("detected multivariate dataset")
        X = df.drop(["timestamp", "is_anomaly"], axis=1).values
        print(X.shape)
        # Train the autoencoder model
        autoencoder = MultivariateAutoencoder(
            X, num_features=X.shape[1], encoding_dim=5
        )
        autoencoder.fit()

        # Predict the reconstruction error for the input data X
        threshold_start = 0.1
        is_anomaly = _predict_with_threshold(X, threshold_start)
        print("Anomaly indices: ", np.where(is_anomaly))

        # load original data as df
        df = pd.read_csv(
            "datasets/csv_datasets/open-anomaly-detection-benchmark/datasets/" + dir
        )
        # add anomaly column to df
        df["anomaly"] = is_anomaly

        # save df to csv
        df.to_csv(
            "results_junkbox/autoencoder_results/" + dir + "_autoencoder.csv",
            index=False,
        )

        # save plot of anomaly
        fig, ax = plt.subplots(figsize=(30, 6))
        # ax.plot(df["timestamp"], df["value"], color="gray")
        # actual anomalies vertical lines
        for idx in np.where(df["is_anomaly"])[0]:
            print(idx)
            print("found!")
            ax.axvline(df["timestamp"][idx], color="red", alpha=0.5)
        # predicted anomalies vertical lines
        for idx in np.where(df["anomaly"])[0]:
            ax.axvline(df["timestamp"][idx], color="blue", alpha=0.5)
        plt.title(dir + " autoencoder anomaly detection")

        # save plot
        plt.savefig("results_junkbox/autoencoder_results/" + dir + "_autoencoder.png")


if __name__ == "__main__":
    dir = input("Enter the subdir of the data: ")
    main(dir)
