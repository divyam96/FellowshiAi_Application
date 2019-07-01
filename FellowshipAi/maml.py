from collections import OrderedDict

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import nn

from maml_cnn import MamlCNN
from util import OmniglotDataset, NShotTaskSampler, create_nshot_task_label, categorical_accuracy


class OmniglotMamlKwayNshot:

    def __init__(self, k_way, n_shot, num_queries, num_tasks_per_batch, path=None):
        """
        Class that defines the necessary models and functions to train the meta learner on the
        Omniglot data set using the MAML(Model-Agnostic Meta-Learning ) approach.

        :param k_way: int Number of possible classes in each query
        :param n_shot: int Number of support data points for training each class
        :param num_queries: int Number of queries for each class
        :param num_tasks_per_batch: int Number of tasks for each batch in data set.
        :param path: str Absolute path to an existing model(optional)
        """

        self.k = k_way
        self.n = n_shot
        self.q = num_queries
        self.num_tasks = num_tasks_per_batch
        self.device = self.get_device_type()
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.epoch = 0

        self.__meta_learner = self.load_model(k_way=self.k, path=path)
        self.__meta_learner.to(self.device, dtype=torch.double).train()
        self.__meta_optimiser = torch.optim.Adam(self.__meta_learner.parameters(), lr=0.001)
        self.__results = self._init_results_dict()

    def fit(self, tasks):
        """

        :param tasks: Tensor containing the tasks the meta learner needs to be trained upon.
        :return:
        """
        task_losses = []

        for task in tasks:
            x_fast_train, x_meta_train = self._sample_data(task)

            fast_weights = self._train_fast_model(x_fast_train, epochs=1, create_graph=True)

            y = create_nshot_task_label(self.k, self.q).to(self.device)
            logits = self.__meta_learner.functional_forward(x_meta_train, fast_weights)
            loss = self.loss_fn(logits, y)
            task_losses.append(loss)

        self.__meta_learner.train()
        self.__meta_optimiser.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_loss.backward()
        self.__meta_optimiser.step()
        self.__results["train"][self.epoch] += meta_batch_loss.item()

    def eval(self, tasks):
        """

        :param tasks: Tensor containing the tasks the meta learner needs to be evaluated upon.
        :return:
        """
        y_softmax_batch = []

        for task in tasks:
            support_data, query_data = self._sample_data(task)

            fast_weights = self._train_fast_model(x_fast_train=support_data, epochs=1, create_graph=False)

            logits = self.__meta_learner.functional_forward(query_data, fast_weights)
            y_softmax = logits.softmax(dim=1)
            y_softmax_batch.append(y_softmax)

        y_softmax_batch_tensor = torch.cat(y_softmax_batch)
        return y_softmax_batch_tensor


    def run_epoch(self, train_batches, val_batches):
        """
        Primary function that defines an epoch.
        An epoch involves:
         1) Training step responsible for training the meta learner.
         2) Validation step responsible for measuring the performance of the trained model

        :param train_batches: DataLoader object responsible for generating training batches
        :param val_batches: DataLoader object responsible for
        :return:
        """
        self._init_epoch()

        self.fit_model(train_batches) # Training step
        self.evaluate_model(val_batches) # Validation step

        self._publish_epoch_results()

    def fit_model(self, train_batches):
        """
        Function to fit the meta learner on batches of training tasks.

        :param train_batches: DataLoader object responsible for generating training batches
        :return:
        """

        for batch_index, batch in enumerate(train_batches):
            training_tasks, y = self._get_tasks(batch)
            self.fit(training_tasks)

    def evaluate_model(self, val_batches):
        """
        Function to evaluate the meta learner on batches of validation tasks.

        :param val_batches: DataLoader object responsible for generating validation batches
        :return:
        """
        for batch_index, batch in enumerate(val_batches):
            validation_tasks, y = self._get_tasks(batch)
            y_softmax_batch = self.eval(validation_tasks)
            self._update_batch_accuracies(batch_index, y, y_softmax_batch)
        self._update_total_accuracies()


    def _train_fast_model(self, x_fast_train, epochs, create_graph):
        """
        Function to train the fast model on the support data points

        :param x_fast_train: k*n Tensor of support data points
        :param epochs: int number of epochs to train the fast model for
        :param create_graph: bool defines wheteher second order derivatives will be involved
        :return:
        """

        fast_weights = OrderedDict(self.__meta_learner.named_parameters())

        for inner_batch in range(epochs):
            y = create_nshot_task_label(self.k, self.n).to(self.device)
            logits = self.__meta_learner.functional_forward(x_fast_train, fast_weights)
            loss = self.loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - 0.4 * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        return fast_weights

    def _sample_data(self, task):
        """
        Function to sample a task into support and query data.

        :param task: Tesnsor that contains the support and query data for each internal training/evaluation step.
        :return:
        """
        support_data = task[:self.n * self.k]
        query_data = task[self.n * self.k:]
        return support_data, query_data

    def _get_tasks(self, batch):
        """
        Transform a batch of sorted images into tasks where each task contains k*n support data points
        and k*q query data points.

        :param batch: Tensor that defines a batch as returned by the DataLoader
        :return: list of tasks, list of labels
        """
        x, y = batch
        x = x.reshape(self.num_tasks, self.n * self.k + self.q * self.k, 1, x.shape[-2], x.shape[-1])
        x = x.double().to(self.device)
        y = create_nshot_task_label(self.k, self.q).to(self.device).repeat(self.num_tasks)
        return x, y

    def _init_results_dict(self):
        """

        :return: dict Empty dictionary to record the performance of each epoch
        """
        results = dict(train=OrderedDict(),
                       accuracy=OrderedDict(),
                       current_batch_accuracies=OrderedDict())
        return results

    def _init_epoch(self):
        """
        Prepare the results dictionary for the upcoming epoch

        :return:
        """
        self.__results["train"][self.epoch] = 0
        self.__results["accuracy"][self.epoch] = 0
        self.__results["current_batch_accuracies"] = OrderedDict()
        print(f"-----------Epoch{self.epoch}-----------")

    def _update_batch_accuracies(self, batch_index, y, y_softmax_batch):
        """

        :param batch_index: int batch index to identify the batch number
        :param y: Tensor containng the class number each data point in the batch belongs to.
        :param y_softmax_batch: Tensor softmax confidence levels for each class
        :return:
        """
        self.__results["current_batch_accuracies"][batch_index] = categorical_accuracy(y, y_softmax_batch)

    def _update_total_accuracies(self):
        """
        Function to record validation accuracies for the current epoch.
        :return:
        """
        self.__results["accuracy"][self.epoch] = sum(self.__results["current_batch_accuracies"].values())/\
                                                     len(self.__results["current_batch_accuracies"].values())


    def _publish_epoch_results(self):
        """
        Function that handles the post epoch tasks like saving models and publishing epoch results to stdout

        :return:
        """
        accuracy = self.__results["accuracy"][self.epoch]
        torch.save(self.__meta_learner.state_dict(), f'/content/models/maml/epoch{self.epoch}_accuracy{accuracy}.pth')
        print("Total train loss:", self.__results["train"][self.epoch])
        print("Validation accuracy: ", self.__results["accuracy"][self.epoch])
        self.epoch += 1

    def plot_training_loss(self):
        """
        Function to plot training loss vs epochs

        :return:
        """
        plt.plot(self.__results["train"].keys(), self.__results["train"].values())
        plt.xlabel('epochs')
        plt.ylabel('Training Loss')

    def plot_validation_accuracy(self):
        """
        Function to plot validation accuracy vs epochs

        :return:
        """
        plt.plot(self.__results["accuracy"].keys(), self.__results["accuracy"].values())
        plt.xlabel('epochs')
        plt.ylabel('Validation Accuracy')

    @staticmethod
    def get_device_type():
        """
        Function to detect device type

        :return: device type
        """
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_model(k_way, path=None):
        """
        Function to load a pre existing model into the Meta learner CNN if it exists and create a new one

        :param k_way: Number of classes the model is supposed to discriminate between.
        :param path: Absolute path to an existing model
        :return: MamlCNN neural net that is to be used for the k way nshot classification task
        """
        meta_learner = MamlCNN(num_input_channels=1, k_way=k_way, final_layer_size=64)
        if path:
            meta_learner.load_state_dict(torch.load(path))
        return meta_learner

    @staticmethod
    def load_data(background_datset_path, evaluation_dataset_path):
        """
        Function to load the Omniglot dataset

        :param background_datset_path:
        :param evaluation_dataset_path:
        :return:
        """
        background = OmniglotDataset('background', path=background_datset_path)
        evaluation = OmniglotDataset('evaluation', path=evaluation_dataset_path)
        return background, evaluation

    @staticmethod
    def get_task_batches_from_dataset(dataset, k, n, q, num_tasks, num_batches):
        """

        :param dataset: Omniglot Dataset that is to be broken down into batches
        :param k: int number of classes
        :param n: int number of support data points for each class
        :param q: int number of query data points for each class
        :param num_tasks: int number of tasks per batch
        :param num_batches: number of batches to be generated
        :return: generator that generates batches from the dataset
        """
        batches = DataLoader(dataset,
                             batch_sampler=NShotTaskSampler(dataset, num_batches, n=n, k=k, q=q, num_tasks=num_tasks),
                             num_workers=8)
        return batches

