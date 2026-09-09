# iris-classifier (Decision Tree)

## Overview

An end-to-end machine learning example built for the AI Fundamentals course
assessment. It trains a decision tree classifier on the classic Iris dataset
using scikit-learn, then reports accuracy and a confusion matrix.

The dataset has three flower species, so this is a multiclass classification problem. The data is loaded directly from scikit-learn, so no download is
needed.

## Quick start

```bash
git clone https://github.com/Mohsenalghasi/iris-classifier.git
cd iris-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

The script accepts two optional arguments:

```bash
python src/train.py --test-size 0.2 --random-state 42
```

## Output

The script prints the accuracy and the confusion matrix in the terminal, and then saves the confusion matrix plot as a .png file in `outputs/`.

Result:

```
Decision tree accuracy: 1.000
Confusion matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

## Notebook

`notebooks/iris_model.ipynb` walks through the same steps one at a time, and ends with a short explanation of the results.

## Tests

```bash
pytest
```

The test calls the training function and checks that accuracy is at least 0.9.

## Project structure

```
iris-classifier/
├── data/                    # empty, Iris is loaded from scikit-learn
├── notebooks/
│   └── iris_model.ipynb     # walk-through notebook
├── src/
│   └── train.py             # reproducible CLI script
├── tests/
│   └── test_train.py        # basic pytest
├── outputs/                 # created automatically (figures)
├── conftest.py              # marks the project root for pytest
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## License

MIT. See [LICENSE](LICENSE).