### A Transformer-based structure-aware model for tackling the traveling salesman problem ###

A Transformer-based structure-aware model for learning to solve the Travelling Salesman Problem (TSP). Training with REINFORCE with greedy rollout baseline.

### Paper

### Cite:

### Dependencies
* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)


### generate data
Training data and validation data are generated on the fly. To generate validation and test data  for TSP:

# TSP20 dataset used for model validation during training
python generate_data.py --tsp_sizes 20 --tag validation --seed 4321
# TSP20 dataset used for model evaluation
python generate_data.py --tsp_sizes 20 --tag test --seed 1234

# TSP20, TSP50, and TSP100 datasets
python generate_data.py --tsp_sizes 20 50 100 --tag validation --seed 4321
python generate_data.py --tsp_sizes 20 50 100 --tag test --seed 1234


### Model training

# Train TSP20 model
python train.py --tsp_size 20 --run_name 'tsp20_training' --val_dataset data/tsp/tsp20_validation_seed4321.pkl
# Train TSP50 model
python train.py --tsp_size 50 --run_name 'tsp50_training' --val_dataset data/tsp/tsp50_validation_seed4321.pkl
# Train TSP100 model
python train.py --tsp_size 100 --run_name 'tsp100_training' --val_dataset data/tsp/tsp50_validation_seed4321.pkl


### Model evaluation
test.py will record tour distances and time consumed into file. The result pkl files are in folder 'results' by default.
The result data file contains a tuple with two elements:
1. A list of tuples, like [(),(),...,()], where each tuple contains three values:(distance, [tour], time)
2. eval_batch_size

# pretrained model
TSP20: pretrained/tsp_20/final-model.pt
TSP50: pretrained/tsp_50/final-model.pt
TSP100: pretrained/tsp_100/final-model.pt

# Test trained model using greedy strategy
python test.py --dataset data/tsp/test/tsp20_testdata_10k.pkl --model pretrained/tsp_20/final-model.pt --decode_strategy greedy
python test.py --dataset data/tsp/test/tsp50_testdata_10k.pkl --model pretrained/tsp_50/final-model.pt --decode_strategy greedy
python test.py --dataset data/tsp/test/tsp100_testdata_10k.pkl --model pretrained/tsp_100/final-model.pt --decode_strategy greedy
# Test trained model using sampling strategy
python test.py --dataset data/tsp/test/tsp20_testdata_10k.pkl --model pretrained/tsp_20/final-model.pt --decode_strategy sample --sample_times 1024
python test.py --dataset data/tsp/test/tsp50_testdata_10k.pkl --model pretrained/tsp_50/final-model.pt --decode_strategy sample --sample_times 1024
python test.py --dataset data/tsp/test/tsp100_testdata_10k.pkl --model pretrained/tsp_100/final-model.pt --decode_strategy sample --sample_times 1024


## Acknowledgements
We would like to express our gratitude to Kool et al. for their original source code, which has been an essential foundation for constructing and implementing the model, as well as the training procedures used in this project. This work would not have been developed without their contributions.
You can find Kool's original source code at: [https://github.com/wouterkool/attention-learn-to-route]

We would also like to thank Chengxuan Ying et al. for their inspiring study. Their source code provided valuable references and insights, which were instrumental in shaping aspects of this project. We referred to parts of their source code during our work.
You can find Chengxuan Ying et al.'s source code at: [https://github.com/microsoft/Graphormer.git]