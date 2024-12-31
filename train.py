import os
import time
import json
import torch
import argparse
import pprint as pp
import torch.optim as optim

from utils import load_problem
from tensorboard_logger import Logger as TbLogger
from train_impl import train_epoch, validate, get_inner_model
from nets.structure_aware_attention_model import SAAttentionModel
from reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline


def run(opts):

    # print out the args
    pp.pprint(vars(opts))

    # Set a random seed
    torch.manual_seed(opts.seed)

    #configure tensorboard
    tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.tsp_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save the configuration of arguments for current trainning
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Specific for TSP problem
    problem = load_problem("tsp")

    # Using Transformer Attention
    model_class=SAAttentionModel
   
    model = model_class(
        # zz ++++++++++++++++
        opts.tsp_size,
        # zz ++++++++++++++++
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        warmupBaseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
 
    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': warmupBaseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(warmupBaseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Initialize learning rate scheduler, decay by lr_decay!
    ## zz
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** ((epoch//opts.lr_decay_start_index)*((epoch-opts.lr_decay_start_index)//opts.lr_decay_per_epochs)))

    # Start the training loop
    ## zz: val_dataset size: 10,000；
    val_dataset = problem.make_dataset(
        size=opts.tsp_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.n_epochs):
            ## zz: impose early stop mechanism
            if epoch -warmupBaseline.baseline.epoch > opts.eary_stop_epochs:
                print("early stop at epoch:{}, baseline epoch is {}".format(epoch,warmupBaseline.baseline.epoch))
                break
            
            train_epoch(
                model,
                optimizer,
                warmupBaseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )
        ## zz: save baseline model as the final model
        torch.save(
            {
                'model': get_inner_model(warmupBaseline.baseline.model).state_dict()
            },
            os.path.join(opts.save_dir, 'final-model.pt')
        )

        print("baseline epoch is {}, it used as the final model, and saved with [final-model.pt]".format(warmupBaseline.baseline.epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure_aware_Transformer_Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Mandatory arguments
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--tsp_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    
    # params of model structure
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    parser.add_argument('--problem', default='tsp', help="Currently only support 'tsp'")

    # Training （optional parameters）
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--n_epochs', type=int, default=200, help='The number of epochs to train')
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay per epoch')
    parser.add_argument('--lr_decay_start_index', type=int, default=100, help='Learning rate decay per epoch')
    parser.add_argument('--lr_decay_per_epochs', type=int, default=2, help='Learning rate decay per epoch')
    parser.add_argument('--eary_stop_epochs', type=int, default=20, help='The number of epochs for early stop')

    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--eval_batch_size', type=int, default=1024,help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    
    parser.add_argument('--val_size', type=int, default=10000, help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_size_increment', type=float, default=0.2, help='percentage of increment instances used for reporting validation performance')
    parser.add_argument('--val_size_max', type=int, default=400000, help='percentage of increment instances used for reporting validation performance')
    parser.add_argument('--val_size_increase_startAt', type=int, default=100, help='the start epoch index of val_size increment')
    
    # optional parameters
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--checkpoint_encoder', action='store_true', help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    parser.add_argument('--log_step', type=int, default=25, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--model', default='attention', help="Model, currently only support 'attention'")
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--exp_beta', type=float, default=0.8, help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout', help="Currently only support 'rollout'")
    parser.add_argument('--bl_alpha', type=float, default=0.05, help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None, help='Data distribution to use during training')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    # end of parameters

    opts =parser.parse_args()

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.tsp_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"

    run(opts)    
