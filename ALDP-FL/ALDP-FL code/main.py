import argparse, json
import datetime
import os
import pandas as pd
import logging
import torch, random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from server import *
from client import *
import models, datasets

def federated_learning(k):
	log_filename = f"{k}.txt"
	CONFIG_FILE_PATH = "./conf.json"
	if not os.path.isfile(CONFIG_FILE_PATH):
		raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE_PATH}")
	with open(CONFIG_FILE_PATH, 'r') as f:
		conf = json.load(f)

	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

	server = Server(conf, eval_datasets)
	clients = []

	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, k, c))

	num_epochs = conf["global_epochs"]
	#acc_list = []
	#loss_list = []
	with open(log_filename, 'w') as log_file:
		log_file.write("Epoch,Accuracy\n")
	for epoch in range(num_epochs):
		print("Epoch {}/{}".format(epoch, num_epochs - 1))
		print("-" * 30)
		candidates = random.sample(clients, conf["c"])

		weight_accumulator = {}

		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		for candidate in candidates:
			diff = candidate.local_train(server.global_model)

			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])

		server.model_aggregate(weight_accumulator)
		acc = server.model_eval()
		with open(log_filename, 'a') as log_file:
			log_file.write(f"{epoch},{acc:.4f}\n")
		#acc_list.append(acc)
		#loss_list.append(loss)
		print("acc:",acc)
	#fl_process = pd.DataFrame(data={"epoch": range(num_epochs),
									#"acc": acc_list,
									#"loss":loss_list})

	#return fl_process

def plot_data_from_files(*file_paths):


    plt.figure(figsize=(8,6))

    for file_path in file_paths:
        try:
            data = pd.read_csv(file_path)
            if 'Epoch' in data.columns and 'Accuracy' in data.columns:
                plt.plot(data['Epoch'], data['Accuracy'], color='black',label=file_path)
            else:
                print(f"file {file_path} missing 'Epoch' or 'Accuracy'")
        except Exception as e:
            print(f"Unable to read the file {file_path}，error: {e}")

    #plt.title('Training Accuracy from Multiple Files')
    plt.xlabel('Global Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
	for k in range(2,5):
		federated_learning(k)
	plot_data_from_files("2.txt","3.txt""4.txt","5.txt")






				
			
		
		
	
		
		
	