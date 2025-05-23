from collections import deque
from scipy.stats import truncnorm

import numpy as np
from fontTools.misc.bezierTools import epsilon

import models, torch, copy
import math


class Client(object):

    def __init__(self, conf, model, train_dataset, k, id=-1):

        self.conf = conf

        self.local_model = models.get_model(self.conf["model_name"])

        self.k = k

        self.client_id = id

        self.dp = True

        self.train_dataset = train_dataset

        # Divide the training data evenly according to the number of clients
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        # Divide the training data of each client into bundles and sample randomly
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())  # Copy the incoming model

        # print("\n\nlocal model train ... ... ")
        # for name, layer in self.local_model.named_parameters():
        #	print(name, "->", torch.mean(layer.data))

        # print("\n\n")
        optimizer = torch.optim.Adam(
            self.local_model.parameters(),  # Model parameters that need to be optimized
            lr=self.conf['lr'],  # Learning rate
            eps=1e-8,  # Prevent division by zero
            weight_decay=self.conf.get('weight_decay', 0),  # L2 Regularization
            amsgrad=False
        )
        #optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    #momentum=self.conf['momentum'])

        history_norms = {
            name: deque(maxlen=self.k)
            for name, _ in self.local_model.named_parameters()
        }  # Historical comparison of standard products
        epoch = self.conf["local_epochs"]
        delta = 10e-5
        e_i = 3
        o = 20
        #print("ei:", e_i)
        B = (math.exp(e_i) - o) / (math.exp(e_i) + o)
        #print("B:", B)
        for e in range(epoch):
            #print("\n\n")
            #print("epoch:",e)
            previous_model = copy.deepcopy(self.local_model.state_dict())
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                self.local_model.train()
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()

                optimizer.step()

            current_model = self.local_model.state_dict()

            if self.dp:
                #print("previous:", previous_model)
                #print("current:", current_model)

                layer_diff = models.layer_norm2(current_model, previous_model) #求每层的更新量
                #print("diff:",layer_diff)
                avg_l2_norm = dict()
                for name, l2_norm in layer_diff.items():
                    if name in history_norms: #Some layers do not participate in the update
                        history_norms[name].append(l2_norm)
                        avg_l2_norm[name] = sum(history_norms[name]) / len(history_norms[name])  # The cropping threshold of each layer
                    #print("history:",history_norms[name])
                    #print("name:",name,"sum:",sum(history_norms[name]),"cliped:",avg_l2_norm[name])
                for name, layer in self.local_model.named_parameters():
                    ldiff = torch.norm(layer.data - previous_model[name], 2).item()
                    #print(name,"diff:",ldiff)
                    #print("name:", name, "cliped:", avg_l2_norm[name])
                    if ldiff > avg_l2_norm[name]:
                        clipped_difference = (avg_l2_norm[name] / ldiff) * (layer.data - previous_model[name]) #"Cut"
                        #print("previous_model:", layer.data - previous_model[name])
                        #print("clipped_difference:",clipped_difference)
                    else:
                        clipped_difference = layer.data - previous_model[name]
                        # print("cliped:",avg_l2_norm[name])
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * avg_l2_norm[name] / e_i
                    #print("sigma:",sigma)
                    min_val = -B
                    max_val = B
                    noise = truncnorm.rvs(min_val, max_val, loc=0, scale=sigma, size=layer.size())
                    noise = torch.tensor(noise, dtype=torch.float32).cuda()
                    #print("data:",clipped_difference)
                    #print("\n")
                    #print("noise:",noise)
                    layer.data.copy_(previous_model[name] + clipped_difference + noise) #更新参数
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])

        # print("\n\nfinishing local model training ... ... ")
        # for name, layer in self.local_model.named_parameters():
        #	print(name, "->", torch.mean(layer.data))
        return diff