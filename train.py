import sys
import data
from options.train_option import TrainOptions
# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
# dataloader = data.create_dataloader(opt)

