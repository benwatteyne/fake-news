import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from features import FeatureEmbeddings
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tqdm.notebook import tqdm

import sklearn
import numpy as np

import sys


class Runner:
    def __init__(self, data, test_data):
        self.data = data.sample(frac=1)  # this is where your randomize.
        self.test_data = test_data
        # sys.stdout.write()
        self.data.head()
        self.train_df = DataFrame()
        self.test_df = DataFrame()
        self.embeddings = FeatureEmbeddings()
        self.val_loader = DataLoader
        self.model = Network(len(data) - 1, 2)
        self.batch_size = 30



    def load_embeddings(self, article_col, url_col, header_col, target_col):
        self.embeddings.create(self.data, article_col=article_col, url_col=url_col,
                          header_col=header_col)
        self.embeddings.features['target'] = self.data['label'].reset_index(drop=True)
        # self.embeddings.features = self.embeddings.features.sample(frac = 1)
        splt = int(len(self.embeddings.features) * 7.5 / 10.0)
        self.train_df = self.embeddings.features.iloc[:splt, :]
        self.test_df = self.embeddings.features.iloc[splt:, :]

    def train_model(self):
        train_dataset = FakeNewsDataset(self.train_df)
        val_dataset = FakeNewsDataset(self.test_df)

        num_epochs = 150
        batch_size = 30

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = Network(len(self.embeddings.features.columns) - 1, 2)
        # Use the code below to load
        path = "models/fake-news-classifier.pt"
        # model.load_state_dict(torch.load(path))

        objective = torch.nn.CrossEntropyLoss()  # loss function
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Run your training / validation loops

        train_losses_avgs = []
        validate_losses_avgs = []

        train_loop = tqdm(total=len(train_loader) * num_epochs, position=0)  # the little progress bar thing
        validate_loop = tqdm(total=len(self.val_loader) * num_epochs, position=0)

        for epoch in range(num_epochs):

            train_losses = []

            for x, y_truth in train_loader:
                optimizer.zero_grad()  # forget about the gradient you computed last time

                y_hat = model(x)
                loss = objective(y_hat, y_truth)

                train_losses.append(loss)

                train_loop.set_description('Training loss: {:.4f}'.format(loss.item()))
                train_loop.update(1)

                loss.backward()  # computes the gradient and stores it in the variable

                optimizer.step()

            train_losses_avgs.append(sum(train_losses) / len(train_losses))

            validate_losses = []

            for x, y_truth in self.val_loader:
                y_hat = model(x)
                loss = objective(y_hat, y_truth)

                validate_losses.append(loss)

                validate_loop.set_description('Validation loss: {:.4f}'.format(loss.item()))
                validate_loop.update(1)

            validate_losses_avgs.append(sum(validate_losses) / len(validate_losses))

        train_loop.close()
        validate_loop.close()

        plt.plot(train_losses_avgs, label='Training')
        plt.plot(validate_losses_avgs, color='orange', label='Validation')
        plt.xlabel('Time (Iterations)')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        self.model = model
        return model

    def test(self, model):
        correct = 0
        total = 0
        y_hat = []
        y_true = []
        with torch.no_grad():
            for x, targets in self.val_loader:
                prediction = model(x)
                _, predicted = torch.max(prediction.data, 1)
                for pred, target in zip(predicted, targets):
                    y_hat.append(int(pred))
                    y_true.append(int(target))
                    total += 1
                    if pred == target:
                        correct += 1

        cm = confusion_matrix(y_true, y_hat)
        tn, fp, fn, tp = cm.ravel()
        # recall = true positives / (true positives + false negatives)
        # precision = true positives / (true positives + false positives)
        print('Accuracy:', correct / total)
        print('Recall:', tp / (tp + fn))
        print('Precision:', tp / (tp + fp))

        df = pd.DataFrame(cm, index=['Real', 'Fake'], columns=['Pred. Real', 'Pred. Fake'])
        sns.heatmap(df, square=True, annot=True, vmin=0, fmt="d")
        plt.show()

    def understand(self):
        class_names = ['true', 'fake']  # 0 is a real news source, 1 is fake news

        from lime.lime_text import LimeTextExplainer
        text_explainer = LimeTextExplainer(class_names=class_names)

        idx = 7 # **actually, this is decided up in the LIME Tabular section ^^^
        txt = self.data.iloc[idx]['text']  # take the text out of our example document
        texp = text_explainer.explain_instance(txt, classifier_fn=self.predict_text, num_features=20)  # get an explanation

        print('Document id: %d' % idx)
        # print(subset.loc[idx]['original_article_text_phase2'])
        print("this article is ", class_names[self.data.iloc[idx]['label']])
        print("the model says: ", self.predict_text([txt]))

        texp.show_in_notebook()

    def predict(self, ra):
        row = pd.DataFrame(ra, columns=self.test_df.columns)
        dataset = FakeNewsDataset(row)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        predictions = []
        for x, targets in loader:
            prediction = self.model(x)
            #         print(prediction.data)
            #         print('space')
            predicted = prediction.data
            #         predicted = nn.slice(prediction.data, [1,0], [1, len(prediction.data)])
            #         predicted = torch.sum(prediction.data, 1)
            #         print(predicted)
            #         predicted = prediction.data
            #         print(predicted.item())
            for pred, target in zip(predicted, targets):
                #             print(pred)
                #             print(pred[1])
                #             y_hat.append(int(pred))
                #             y_true.append(int(target))
                #             total += 1
                #             if pred == target:
                #                 correct += 1
                predictions.append([pred[0], pred[1]])
        return np.array(predictions)

    def predict_text(self, txt_lst):
        sset = pd.DataFrame()
        for txt in txt_lst:
            nxt = self.data.iloc[1]
            nxt.at['text'] = txt
            sset = sset.append([nxt])
        embeddings = FeatureEmbeddings()
        embeddings.create(sset, article_col="text", url_col=None,
                               header_col="title")
        embeddings.features['target'] = sset['label'].reset_index(drop=True)
        embeddings.features = embeddings.features.sample(frac=1)

        # only_text = embeddings.features.loc[:,
        #             [x for x in embeddings.features.columns if '_vec_' in x or 'target' in x]]

        ra = embeddings.features.to_numpy()
        return self.predict(ra)


class Network(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()

        self.hidden1 = nn.Linear(in_dim, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, out_dim)

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


class FakeNewsDataset(Dataset):
    def __init__(self, df):
        self.data = df.drop(columns=['target'])
        self.targets = df['target'].astype(int)

    def __getitem__(self, i):
        x = torch.tensor(self.data.iloc[i]).float()
        y = torch.tensor(self.targets.iloc[i]).long()
        return x, y

    def __len__(self):
        return len(self.data)


