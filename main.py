import gc
import re
import pandas as pd
import torch
import chess
import numpy as np

letter2num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num2letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}


# data representation
def board2rep(board):
    pieces = ['p', 'k', 'b', 'r', 'q', 'k']
    layers = []

    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep


def create_rep_layer(board, type):
    s = str(board)
    s = re.sub(f'[^{type}{type.upper()} \n]', ".", s)
    s = re.sub(f"{type}", "-1", s)
    s = re.sub(f"{type.upper()}", '1', s)
    s = re.sub(f'\.', "0", s)
    board_mat = []
    for row in s.split("\n"):
        row = row.split(" ")
        row = [int(x) for x in row]
        board_mat.append(row)
    return np.array(board_mat)


def move2rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    # create a 8x8 empty matrix
    from_output_layer = np.zeros((8, 8))
    from_row = 8 - int(move[1])
    from_column = letter2num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8, 8))
    to_row = 8 - int(move[3])
    to_column = letter2num[move[2]]
    to_output_layer[to_row, to_column] = 1
    return np.stack([from_output_layer, to_output_layer])


def create_move_list(s):
    return re.sub("\d*\. ", '', s).split(' ')[:-1]


# data load
chess_data_raw = pd.read_csv("chess_games.csv", usecols=["AN", "WhiteElo"])
chess_data = chess_data_raw[chess_data_raw["WhiteElo"] > 2000]
del chess_data_raw
gc.collect()
chess_data = chess_data[["AN"]]
chess_data = chess_data[~chess_data["AN"].str.contains("{")]
chess_data = chess_data[chess_data["AN"].str.len() > 20]

print(chess_data.shape)

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F


class ChessDataset(Dataset):
    def __init__(self, games):
        super(ChessDataset, self).__init__()
        self.games = games

    def __len__(self):
        return 40_000

    def __getitem__(self, item):
        game_i = np.random.randint(self.games.shape[0])
        random_game = chess_data["AN"].values[game_i]
        moves = create_move_list(random_game)
        game_state_i = np.random.randint(len(moves) - 1)
        next_move = moves[game_state_i]
        moves = moves[:game_state_i]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        x = board2rep(board)
        y = move2rep(board)
        if game_state_i % 2 == 1:
            x *= -1
        return x, y


# connected CNN neural network
data_train = ChessDataset(chess_data["AN"])
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)


class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = x + x_input
        x = self.activation2(x)
        return x


class ChessNet(nn.Module):

    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_layer = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)
        return x


def checkmate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()


def distribution_over_moves(vals):
    probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs **= 3
    probs = probs / probs.sum()
    return probs


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = ChessNet().to(device=device)

num_classes = 10
learning_rate = 0.001
batch_size = 8
num_epochs = 1
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(DataLoader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # flatten matrix
        # data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        # grad descent
        optimizer.step()


def choose_moves(board, player, color):
    legal_moves = list(board.legal_moves)

    move = checkmate_single(board)
    if move is not None:
        return move
    x = torch.Tensor(board2rep(board)).float().to("cuda0")
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)
    move = model.module.predict(x)
    vals = []

    return move
