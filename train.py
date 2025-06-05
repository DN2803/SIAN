import sys
import data
from options.train_option import TrainOptions
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

