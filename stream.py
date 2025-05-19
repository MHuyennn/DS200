import os
import json
import time
import socket
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Streams HandWritten_Character dataset to Spark')
parser.add_argument('--folder', '-f', help='Dataset folder (train/test)', required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=False, type=int, default=64)
parser.add_argument('--sleep', '-t', help='Streaming interval (in seconds)', required=False, type=int, default=3)
parser.add_argument('--endless', '-e', help='Stream endlessly', action='store_true')

TCP_IP = "localhost"
TCP_PORT = 8000

class DatasetStreamer:
    def __init__(self, image_size=(32, 32)) -> None:
        self.image_size = image_size
        self.class_to_idx = {}
        self.epoch = 0

    def build_class_mapping(self, folder_path):
        classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789') + ['@', '#', '$', '&']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.class_to_idx['0'] = self.class_to_idx['O']
        for folder in os.listdir(folder_path):
            if folder not in self.class_to_idx:
                print(f"[WARNING] Unknown folder {folder} in {folder_path}")

    def load_batches(self, folder_path, batch_size):
        data, labels = [], []
        for class_name in sorted(os.listdir(folder_path)):
            if class_name in self.class_to_idx:  
                class_folder = os.path.join(folder_path, class_name)
                label = self.class_to_idx[class_name]
                for img_file in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_file)
                    img = Image.open(img_path).convert("L").resize(self.image_size)
                    img_np = np.array(img).astype(np.float32) / 255.0 
                    img_np = img_np.reshape(-1)
                    data.append(img_np)
                    labels.append(label)
        
        # Batching
        batches = []
        total = len(data)
        for i in range(0, total, batch_size):
            batch_images = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batches.append((batch_images, batch_labels))
        return batches

    def connectTCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"[TCP] Waiting for connection on port {TCP_PORT}...")
        conn, addr = s.accept()
        print(f"[TCP] Connected to {addr}")
        return conn

    def stream(self, tcp_conn, folder_path, batch_size, sleep_time, endless=False):
        self.build_class_mapping(folder_path)

        while True:
            batches = self.load_batches(folder_path, batch_size)
            pbar = tqdm(total=len(batches), desc=f"Epoch {self.epoch}")
            for batch_images, batch_labels in batches:
                payload = {}
                for i, img in enumerate(batch_images):
                    payload[i] = {f"feature{j}": float(val) for j, val in enumerate(img)}
                    payload[i]['label'] = int(batch_labels[i])

                try:
                    tcp_conn.send((json.dumps(payload) + "\n").encode())
                except Exception as e:
                    print("[ERROR] Failed to send batch:", e)
                    return
                time.sleep(sleep_time)
                pbar.update(1)

            self.epoch += 1
            if not endless:
                break

if __name__ == '__main__':
    args = parser.parse_args()
    folder_path = args.folder
    batch_size = args.batch_size
    sleep_time = args.sleep
    endless = args.endless

    streamer = DatasetStreamer()
    tcp_conn = streamer.connectTCP()
    streamer.stream(tcp_conn, folder_path, batch_size, sleep_time, endless)
    tcp_conn.close()