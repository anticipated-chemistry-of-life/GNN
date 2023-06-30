# GNN
In this repo we try to implement a graph neural network that should ideally predict the chemical compositions of organisms in across the tree of life. 

If the user wants to reproduce this, it should first download the latest version of LOTUS [here](https://zenodo.org/record/7534071) or like this : 

```bash
wget https://zenodo.org/record/7534071/files/230106_frozen_metadata.csv.gz
```

The algorithm implements a link classification task in a graph between nodes `species` and nodes `molecule`. We use HinSAGE with mean aggregator from [StellarGraph](https://stellargraph.readthedocs.io/en/stable/index.html) library.

## Playing around
To reproduce the model, the user should first : 

```bash
conda create -f environment.yaml
conda activate stellar_graph
```

Then we create a graph of LOTUS and split them into training and testing dataset:
```bash
python graph_creation_train.py
python graph_creation_test.py
```

After grid searching for the best parameters (still in research ), we set the neural network with two hidden layers of 1024 neurons each with activations "elu" and "selu" respectively. The training of the model can be seen in the `HinSAGE_mol_to_species.ipynb` notebook. Testing on unseen data is in `HinSAGE_test.ipynb` notebook

If we want to recreate the entire LOTUS database as a graph simply run : 
```python
g_train = nx.read_graphml("./graph/train_graph.gml")
g_test = nx.read_graphml("./graph/test_graph.gml")
g = nx.compose(g_train, g_test)
```

## Performance
So far we achieve an accuracy of 0.92 on unseen data (with $p<0.5 \rightarrow 0$ (absent) and $p>0.5 \rightarrow 1$ (present) ). 

## 