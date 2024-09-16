from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time

from utils import load_data, accuracy
from models import GAT, SpGAT

# Inference settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA inference.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved model checkpoint.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model initialization
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                  nhid=args.hidden, 
                  nclass=int(labels.max()) + 1, 
                  dropout=args.dropout, 
                  nheads=args.nb_heads, 
                  alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)

# Load the model checkpoint
model.load_state_dict(torch.load(args.checkpoint))

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

# Inference with timing
model.eval()

# Initialize total time and timing for every 10 samples
total_start_time = time.time()
sample_start_time = time.time()

output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])

# Calculate timing every 200 samples
batch_size = 200
for i in range(0, idx_test.size(0), batch_size):
    batch_output = output[i:i + batch_size]
    batch_labels = labels[i:i + batch_size]
    
    batch_loss = F.nll_loss(batch_output, batch_labels)
    batch_acc = accuracy(batch_output, batch_labels)

    # Calculate time for this batch
    batch_end_time = time.time()
    print("Batch {}-{} results: loss= {:.4f}, accuracy= {:.4f}, time= {:.4f}s".format(
        i, min(i + batch_size, idx_test.size(0)), 
        batch_loss.item(), 
        batch_acc.item(), 
        batch_end_time - sample_start_time
    ))
    
    # Reset timer for next batch
    sample_start_time = time.time()

# Total elapsed time
total_end_time = time.time()

print("\nOverall Inference results:")
print("Test loss: {:.4f}".format(loss_test.item()))
print("Test accuracy: {:.4f}".format(acc_test.item()))
print("Total time elapsed for inference: {:.4f}s".format(total_end_time - total_start_time))
