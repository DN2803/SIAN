# train_sian.py

import sys
from collections import OrderedDict
from options.train_option import TrainOptions
import data
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

# parse options
opt = TrainOptions().parse()
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

print(len(dataloader))


# create trainer (model đã là SIAN bên trong)
trainer = Pix2PixTrainer(opt)

# iteration + visualization
iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        try:
            # Train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # Train discriminator
            trainer.run_discriminator_one_step(data_i)

        except Exception as e:
            print(f"❌ Error in training step: {str(e)}")
            raise

        # Visualize
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([
                ('input_label', data_i['label']),
                ('synthesized_image', trainer.get_latest_generated()),
                ('real_image', data_i['image'])
            ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print(f"💾 Saving latest model (epoch {epoch}, steps {iter_counter.total_steps_so_far})")
            trainer.save('latest')
            iter_counter.record_current_iter()

    # update LR
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(f"💾 Saving model at end of epoch {epoch}")
        trainer.save('latest')
        trainer.save(epoch)

print('✅ Training finished successfully.')
