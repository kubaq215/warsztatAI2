import mnist
import numpy as np
from sudoku import Sudoku
from PIL import Image
from matplotlib import colormaps as cm
from tqdm import tqdm
import sys

# get number image from MNIST dataset
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# digit to image
def get_number_img(num, background=255):
  idxs = np.where(test_labels==num)[0]
  idx = np.random.choice(idxs, 1)
  img = test_images[idx].reshape((28, 28)).astype(np.int32)
  img = background - img # make background white
  return img


def create_empty_board():
  cell_size = 28
  rows, cols = 9, 9
  image_size = cell_size * rows + 10
  img_array = np.full((image_size, image_size), 255, dtype=np.uint8)

  i_add = 0
  for i in range(1, rows):
      img_array[cell_size*i+i-1+i_add, :] = 0
      img_array[:, cell_size*i+i-1+i_add] = 0
      
      if i % 3 == 0:
        img_array[cell_size*i+i+i_add, :] = 0
        img_array[:, cell_size*i+i+i_add] = 0
        i_add += 1
  
  return img_array
  
  
# create board image with grid lines
def create_board(puzzle_arr, empty_board):
  i_add = 0
  for i in range(9):
    i_add += 1 if i != 0 and i % 3 == 0 else 0
    j_add = 0
    for j in range(9):
      j_add += 1 if j != 0 and j % 3 == 0 else 0
      num = puzzle_arr[i][j]
      
      if num is not None:
        num_img = get_number_img(num)
        cmap = cm.get_cmap('gray')
        img_gray_array = cmap(num_img/255)
        img_gray_array = (img_gray_array[..., 0] * 255).astype(np.uint8)
        left = i*28 + i + i_add
        right = (i+1)*28 + i + i_add
        top = j*28 + j + j_add
        bottom = (j+1)*28 + j + j_add      
        empty_board[left:right, top:bottom] = img_gray_array
  return Image.fromarray(empty_board)

base = create_empty_board()

for i in tqdm(range(1)):
  seed = np.random.randint(0, sys.maxsize)
  puzzle = Sudoku(3, seed=seed).difficulty(0.5)
  solution = puzzle.solve()
  puzzle_img = create_board(puzzle.board, base.copy())
  puzzle_img.save(f'dataset/images/puzzles/board_{i}.png')
  solution_img = create_board(solution.board, base.copy())
  solution_img.save(f'dataset/images/solutions/board_{i}.png')
  np.save(f'dataset/arrays/puzzles/board_{i}.npy', puzzle.board)
  np.save(f'dataset/arrays/solutions/board_{i}.npy', solution.board)
