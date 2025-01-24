# Requirements

- Python 3.9 or higher

Heavily inspired by [Prolog-Sudoku](https://github.com/barjin/Prolog-Sudoku).

Dataset available at `warsztatAI2/mnist_sudoku_generator/dataset_compressed/`.
You have to unpack the things inside. If you want the full image arrays, then generate it yourself with `warsztatAI2/mnist_sudoku_generator/main.py`.

In the `prolog3.pl` file, you can uncomment various debug prints to get more information about the step-by-step process. There is also a commented-out version attempting to bypass the lack of the `!/0` predicate using `cut/1` (Problog version), but it seems to have no effect.

## Problog Documentation

- [Problog Documentation](https://problog.readthedocs.io/en/latest/)
- [Supported Built-ins and Libraries](https://problog.readthedocs.io/en/latest/prolog.html)

## Thoughts on How It Works

Refer to `warsztatAI2/deepproblog/attempt1/prolog3.pl:300` for detailed insights.