from typing import Iterator
import numpy as np
import os
import torch
from problog.logic import Term, Constant, list2term
from deepproblog.dataset import Dataset
from deepproblog.query import Query
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class MNIST_SudokuNet(nn.Module):
    def __init__(self):
        super(MNIST_SudokuNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        # Only 9 outputs now
        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 9),   # 9 classes => digits 1..9
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.encoder(x)                # 1×28×28 -> ...
        x = x.view(-1, 16 * 4 * 4)         
        x = self.classifier(x)             # -> 9 outputs 
        return x

class SudokuPuzzleDataset(Dataset):
    def __init__(self, subset: str, puzzle_path: str, solution_path: str):
        self.subset = subset
        self.puzzle_files = sorted([f for f in os.listdir(puzzle_path) if f.endswith('.npy')])
        self.puzzles = []
        for fn in self.puzzle_files:
            puzzle = np.load(os.path.join(puzzle_path, fn), allow_pickle=True)
            # Keep puzzle[i][j] == None if empty
            # If not None, we treat it as a 28×28 image
            #   and normalize it (1..255 => /255)
            for i in range(9):
                for j in range(9):
                    if puzzle[i][j] is not None:
                        puzzle[i][j] = puzzle[i][j].astype(np.float32)/255.0
            self.puzzles.append(puzzle)

        self.solution_files = sorted(f for f in os.listdir(solution_path) if f.endswith('.npy'))
        self.solutions = []
        for fn in self.solution_files:
            solution = np.load(os.path.join(solution_path, fn), allow_pickle=True)
            self.solutions.append(solution)

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, item):
        """
        For the Prolog side, we only need to retrieve the actual image data 
        if the cell is not None. We'll handle that in 'to_query' as well.
        """
        # This method is used by the DeepProbLog model to fetch the image.
        # We'll just return a single cell's tensor to illustrate 
        # (not typically used if you do full puzzle queries).
        puzzle_index, row, col = item
        cell = self.puzzles[int(puzzle_index)][int(row)][int(col)]
        if cell is None:
            # Not returning an image for None cells
            raise ValueError("Attempted to retrieve an image for a None cell.")
        # print
        return torch.tensor(cell).unsqueeze(0)  # 1×28×28

    def to_query(self, index: int) -> Query:
        puzzle = self.puzzles[index]
        # We'll create 9 rows of terms
        row_list_terms = []

        for i in range(9):
            col_terms = []
            for j in range(9):
                if puzzle[i][j] is None:
                    # Just an atom 'none'
                    col_terms.append(Constant('none'))
                else:
                    # A reference to the network input
                    image_term = Term("tensor", Term(self.subset, Constant(index), Constant(i), Constant(j)))
                    label = Constant(self.solutions[index][i][j])
                    col_terms.append(Term("digit", image_term, label))

            row_list_terms.append(list2term(col_terms))

        # puzzle_solve( [[...],[...],...] )
        query_term = Term("puzzle_solve", list2term(row_list_terms))
        # print("Query constructed", query_term)
        return Query(query_term)
    
import torch
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.dataset import DataLoader
from deepproblog.train import train_model

network = MNIST_SudokuNet()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# 2) Create the model from 'sudoku_no_clpfd.pl'
model = Model("prolog2.pl", [net])
model.set_engine(ExactEngine(model))

# 3) Load the dataset
train_data = SudokuPuzzleDataset("train", "../../mnist_sudoku_generator/dataset/images/puzzles/train")
model.add_tensor_source("train", train_data)

loader = DataLoader(train_data, batch_size=1, shuffle=False)
train_model(model, loader, 1, log_iter=1)