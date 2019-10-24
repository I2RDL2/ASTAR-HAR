import argparse
from mean_teacher import template

parser = argparse.ArgumentParser(description='Mean teacher')

parser.add_argument('--template', default='vanilla',
                    # choices=['cluster','vanilla','cluster_center','cluster_logits','cluster_class'],
                    help='You can set various templates in template.py')
parser.add_argument('--gpu', type=int, default=0,
                    help='Define which GPUs would be used')

# Data specifications
parser.add_argument('--dataset', type=str, default='Cifar10ZCA',
                    choices=['Cifar10ZCA', 'SVHN','HAR'],
                    help='train dataset name')
parser.add_argument('--dataset_detail', type=str, default=None,
                    help='details of dataset')
parser.add_argument('--n_classes', type=int, default=24,
                    help='number of classes in the dataset')
parser.add_argument('--n_labeled',nargs="*", type=int, default=[1000],
                    help='number of labeled data in totoal dataset')
parser.add_argument('--minibatch_size', type=int, default=100,
                    help='input batch size for training')
parser.add_argument('--n_labeled_per_batch', type=str, default='vary',
                    help='number of labeled data in each batch')
parser.add_argument('--average_labeled',  action='store_true',
                    help='averagely select labeled data from each class')
parser.add_argument('--input_dim', nargs="*", type=int, default=[32,32,3],
                    help='dimension of inputs')
parser.add_argument('--label_dim', nargs="*", type=int, default=[],
                    help='dimension of label')
parser.add_argument('--flip_horizontally', type=bool, default=False,
                    help='horizontally flip input')
parser.add_argument('--translate', type=bool, default=True,
                    help='translate input by up to two pixels')
parser.add_argument('--resize', type=bool, default=False,
                    help='resize input')

# Model specifications
parser.add_argument('--method',type=str, default='mean_teacher_base',choices=['mean_teacher_base','mean_teacher_cluster'],
                    help='mean teacher class to use')
parser.add_argument('--cnn', type=str,default='tower',
                    help='model name')
parser.add_argument('--ckp', type=str, default='.',
                    help='pre-trained weights directory')

# Consistency hyperparameters
parser.add_argument('--ema_consisitency', type=bool, default=True,
                    help='model name')
parser.add_argument('--apply_consistency_to_labeled', type=bool, default=True,
                    help='compute consisitency of labeled samples')
parser.add_argument('--max_consistency_cost', type=float, default=100,
                    help='upper weight of consistency cost')
parser.add_argument('--ema_decay_during_rampup', type=float, default=0.999,
                    help='')
parser.add_argument('--ema_decay_after_rampup', type=float, default=0.999,
                    help='')
parser.add_argument('--consistency_trust', type=float, default=0.0,
                    help='Consistency_trust determines the distance metric to use: mse, kl')
parser.add_argument('--consistency_norm', type=str, default='softmax',
                    choices=['softmax','sigmoid','logits'],
                    help='loss function for consistency loss')

# clusterings
parser.add_argument('--num_cluster_steps', type=int, default=1,
                    help='number of turns to optimize the cluster centers')
parser.add_argument('--consistency_pure_cluster', action='store_true',
                    help='')
# Training specifications
parser.add_argument('--n_runs', type=int, default=5,
                    help='number of runs of different random seeds')
parser.add_argument('--init_run', type=int, default=2000,
                    help='number of runs of different random seeds')
parser.add_argument('--normalize_input', type=bool, default=False,
                    help='normalize input data')
parser.add_argument('--evaluation_span', type=int, default=500,
                    help='do test per every N batches')
parser.add_argument('--training_length', type=int, default=37500,
                    help='number of steps to train')
parser.add_argument('--rampup_length', type=int, default=10000,
                    help='steps of learning rate ramps up to max')
parser.add_argument('--rampdown_length', type=int, default=6250,
                    help='learning rate ramp down in the last steps')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--reset', action='store_true',
                    help='reset the training folder')


# Optimization specifications
parser.add_argument('--optimizer', type=str, default='adam',
                    help='loss function configuration')
parser.add_argument('--max_learning_rate', type=float, default=0.003,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--adam_beta_1_before_rampdown', type=float, default=0.9,
                    help='adam_beta_1_before_rampdown')
parser.add_argument('--adam_beta_1_after_rampdown', type=float, default=0.5,
                    help='ADAM beta1')
parser.add_argument('--adam_beta_2_during_rampup', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--adam_beta_2_after_rampup', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')


# Loss specifications
parser.add_argument('--sig', type=bool, default=False,
                    help='sigmoid or softmax')
parser.add_argument('--losses', nargs="*", type=str, default=['mt_class','mt_cons'],
                    help='all losses to be used')


# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_span', type=int, default=20,
                    help='how many batches to wait before logging training status')

# Pertubation h7perparametesr
parser.add_argument('--input_noise', type=float, default=0.15,
                    help='level of gaussian noise')
parser.add_argument('--student_dropout_probability', type=float,default=0.5,
                    help='')
parser.add_argument('--teacher_dropout_probability', type=float,default=0.5,
                    help='')                    

# Background noise, specially designed for audio dataset
parser.add_argument('--bg_noise', type=bool, default=False,
                    help='whether to have background noise')
parser.add_argument('--bg_noise_input', type=str, default='audio',
                    help='source of background noise')
parser.add_argument('--bg_noise_level', type=float, default=0,
                    help='weight for bg noise input')

# Distilling specifications
parser.add_argument('--max_temp', type=float, default=1,
                    help='max temp during ramp down')
parser.add_argument('--min_temp', type=float, default=1,
                    help='min temp during ramp down')
parser.add_argument('--distilling', action='store_true', default = False,
                    help='using distilling during training')


args = parser.parse_args()
if args.n_labeled_per_batch!='vary':
    args.n_labeled_per_batch = int(args.n_labeled_per_batch)

template.set_template(args)
# args.scale = list(map(lambda x: int(x), args.scale.split('+')))
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

