import math
import os
import numpy as np
import torch
import pickle
import pyspark
import matplotlib.pyplot as plt

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from cnn_model import CNN  # Updated import
from transforms import Transforms

class TrainingConfig:
    num_samples = 5e4
    max_epochs = 10
    learning_rate = 1e-3
    batch_size = 64
    ckpt_interval = 1
    ckpt_interval_batch = 1000
    ckpt_dir = "./checkpoints/"
    model_name = "CNN-HandWritten"
    cache_path = "./DeepImageCache"
    verbose = True

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

class SparkConfig:
    appName = "HandWrittenCharacters"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 8000
    batch_interval = 4

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

from dataloader import DataLoader

class Trainer:
    def __init__(self, model, split: str, training_config: TrainingConfig, spark_config: SparkConfig, transforms: Transforms = Transforms([])) -> None:
        self.model = model
        self.split = split
        self.configs = training_config
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf, self.transforms)
        
        self.accuracy = []
        self.smooth_accuracy = []
        self.loss = []
        self.smooth_loss = []
        self.precision = []
        self.smooth_precision = []
        self.recall = []
        self.smooth_recall = []
        self.f1 = []
        self.cm = np.zeros((39, 39))  # Updated for 39 classes
        self.smooth_f1 = []
        self.epoch = 0
        self.batch_count = 0

        self.test_accuracy = 0
        self.test_loss = 0
        self.test_recall = 0
        self.test_precision = 0
        self.test_f1 = 0

        self.save = True

    def save_checkpoint(self, message):
        path = os.path.join(self.configs.ckpt_dir, self.configs.model_name)
        print(f"Saving Model under {path}/...{message}")
        if not os.path.exists(self.configs.ckpt_dir):
            os.makedirs(self.configs.ckpt_dir)
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.save(f"{path}/accuracy-{message}.npy", self.accuracy)
        np.save(f"{path}/loss-{message}.npy", self.loss)
        np.save(f"{path}/precision-{message}.npy", self.precision)
        np.save(f"{path}/recall-{message}.npy", self.recall)
        np.save(f"{path}/f1-{message}.npy", self.f1)

        np.save(f"{path}/smooth_accuracy-{message}.npy", self.smooth_accuracy)
        np.save(f"{path}/smooth_loss-{message}.npy", self.smooth_loss)
        np.save(f"{path}/smooth_precision-{message}.npy", self.smooth_precision)
        np.save(f"{path}/smooth_recall-{message}.npy", self.smooth_recall)
        np.save(f"{path}/smooth_f1-{message}.npy", self.smooth_f1)

        torch.save(self.model.state_dict(), f"{path}/model-{message}.pth")

    def load_checkpoint(self, message):
        print("Loading Model ...")
        path = os.path.join(self.configs.ckpt_dir, self.configs.model_name)
        self.accuracy = np.load(f"{path}/accuracy-{message}.npy")
        self.loss = np.load(f"{path}/loss-{message}.npy")
        self.precision = np.load(f"{path}/precision-{message}.npy")
        self.recall = np.load(f"{path}/recall-{message}.npy")
        self.f1 = np.load(f"{path}/f1-{message}.npy")

        self.smooth_accuracy = np.load(f"{path}/smooth_accuracy-{message}.npy")
        self.smooth_loss = np.load(f"{path}/smooth_loss-{message}.npy")
        self.smooth_precision = np.load(f"{path}/smooth_precision-{message}.npy")
        self.smooth_recall = np.load(f"{path}/smooth_recall-{message}.npy")
        self.smooth_f1 = np.load(f"{path}/smooth_f1-{message}.npy")

        self.model.load_state_dict(torch.load(f"{path}/model-{message}.pth"))
        print("Model Loaded.")

    def plot(self, timestamp, df: pyspark.RDD) -> None:
        if not os.path.exists("images"):
            os.makedirs("images")
        for i, ele in enumerate(df.collect()):
            image = ele[0].toArray().astype(np.uint8)
            image = image.reshape(32, 32)  # Grayscale image
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.title(f"Training Image {i} | Batch {timestamp}")
            plt.axis('off')
            plt.savefig(f"images/train_image_{timestamp}_{i}.png")
            plt.close()

    def configure_model(self):
        return self.model.configure_model(self.configs)

    def train(self):
        stream = self.dataloader.parse_stream()
        self.raw_model = self.configure_model()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            self.plot(timestamp, rdd)  # Visualize training images
            self.batch_count += 1
            schema = StructType([StructField("image", VectorUDT(), True), StructField("label", IntegerType(), True)])
            df = self.sqlContext.createDataFrame(rdd, schema)
            
            model, predictions, accuracy, loss, precision, recall, f1 = self.model.train(df, self.raw_model)
            
            self.raw_model = model
            self.model = model

            if self.configs.verbose and self.save:
                print(f"Predictions = {predictions}")
                print(f"Accuracy = {accuracy}")
                print(f"Loss = {loss}")
                print(f"Precision = {precision}")
                print(f"Recall = {recall}")
                print(f"F1 Score = {f1}")

            if self.save:
                self.accuracy.append(accuracy)
                self.loss.append(loss)
                self.precision.append(precision)
                self.recall.append(recall)
                self.f1.append(f1)

                self.smooth_accuracy.append(np.mean(self.accuracy))
                self.smooth_loss.append(np.mean(self.loss))
                self.smooth_precision.append(np.mean(self.precision))
                self.smooth_recall.append(np.mean(self.recall))
                self.smooth_f1.append(np.mean(self.f1))

            if self.split == 'train':
                if self.batch_count != 0 and self.batch_count % ((self.configs.num_samples // self.configs.batch_size) + 1) == 0:
                    self.epoch += 1

                if (isinstance(self.configs.ckpt_interval, int) and self.epoch != 0 and 
                    self.batch_count == ((self.configs.num_samples // self.configs.batch_size) + 1) and 
                    self.epoch % self.configs.ckpt_interval == 0):
                    if self.save:
                        self.save_checkpoint(f"epoch-{self.epoch}")
                    self.batch_count = 0
                elif self.configs.ckpt_interval_batch is not None and self.batch_count != 0 and self.batch_count % self.configs.ckpt_interval_batch == 0:
                    if self.save:
                        self.save_checkpoint(f"epoch-{self.epoch}-batch-{self.batch_count}")

            print(f"epoch: {self.epoch} | batch: {self.batch_count}")
            print("Total Batch Size of RDD Received:", rdd.count())
            print("---------------------------------------")

    def predict(self):
        stream = self.dataloader.parse_stream()
        self.raw_model = self.configure_model()
        self.load_checkpoint(self.configs.load_model)
        
        stream.foreachRDD(self.__predict__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __predict__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:     
        if not rdd.isEmpty():
            self.batch_count += 1
            total_batches = math.ceil(1e4 // self.configs.batch_size)
            schema = StructType([StructField("image", VectorUDT(), True), StructField("label", IntegerType(), True)])
            df = self.sqlContext.createDataFrame(rdd, schema)
            
            predictions, accuracy, loss, precision, recall, f1, cm = self.model.predict(df, self.raw_model)
            
            self.cm += cm
            self.test_accuracy += accuracy / total_batches
            self.test_loss += loss / total_batches
            self.test_precision += precision / total_batches
            self.test_recall += recall / total_batches
            self.test_f1 += f1 / total_batches
            print(f"Test Accuracy: {self.test_accuracy}")
            print(f"Test Loss: {self.test_loss}")
            print(f"Test Precision: {self.test_precision}")
            print(f"Test Recall: {self.test_recall}")
            print(f"Test F1 Score: {self.test_f1}")
            print(f"Confusion matrix:\n{self.cm}")

            path = os.path.join(self.configs.ckpt_dir, self.configs.model_name)
            with open(f"{path}/test-scores-{self.configs.batch_size}.txt", "w+") as f:
                f.write(f"Test Accuracy: {self.test_accuracy}\n")
                f.write(f"Test Loss: {self.test_loss}\n")
                f.write(f"Test Precision: {self.test_precision}\n")
                f.write(f"Test Recall: {self.test_recall}\n")
                f.write(f"Test F1 Score: {self.test_f1}\n")
                f.write(f"Confusion matrix:\n{self.cm}\n")
                
            np.save(f"{path}/confusion-matrix.npy", self.cm)

        print(f"batch: {self.batch_count}")
        print("Total Batch Size of RDD Received:", rdd.count())
        print("---------------------------------------")