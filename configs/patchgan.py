import torch
from pytorch_lightning.strategies import DDPStrategy



accelerator = 'gpu'
device = torch.device("cuda") if accelerator=="gpu" else torch.device("cpu")
if accelerator == 'cpu':
    pl_trainer = dict(max_epochs=1000, accelerator=accelerator, log_every_n_steps=50, strategy=DDPStrategy(find_unused_parameters=True), devices=1, sync_batchnorm=True) # CHECK sync_batchnorm in this and below part !!!
else:
    pl_trainer = dict(max_epochs=1000, accelerator=accelerator, log_every_n_steps=50, strategy=DDPStrategy(find_unused_parameters=True), devices=torch.cuda.device_count(), sync_batchnorm=True)  # CHECK strategy and find_unused_parameters!!!

train_batch_size = 4
val_batch_size = 16
test_batch_size = 16

image_size = 256


###########################
##### Dataset Configs #####
###########################

dataset_module = dict(
    num_workers=4,
    train_batch_size=train_batch_size,
    val_batch_size=val_batch_size,
    test_batch_size=test_batch_size,
    train_dataset_config=dict(image_size=256, image_dir_hr="dataset_cropped_hr", image_dir_lr="dataset_cropped_lr", downsample_factor=4,mirror_augment_prob=0.5),
    test_dataset_config=dict(image_size=256, image_dir_hr="dataset_cropped_hr", image_dir_lr="dataset_cropped_lr"),
)

##################
##### Losses #####
##################

loss_dict = dict(
    VGG=dict(weight=1.0),
    Adversarial_G=dict(weight=1.0),
    MSE=dict(weight=1.0),
    Adversarial_D=dict(r1_gamma=10.0, r2_gamma=0.0)
)

#########################
##### Model Configs #####
#########################

super_resolution_module_config = dict(loss_dict=loss_dict, 
    generator_learning_rate=1e-4, discriminator_learning_rate=1e-4, 
    generator_decay_steps=[50_000, 100_000, 150_000, 200_000, 250_000], 
    discriminator_decay_steps=[50_000, 100_000, 150_000, 200_000, 250_000], 
    generator_decay_gamma=0.5, discriminator_decay_gamma=0.5,
    clip_generator_outputs=False,
    use_sed_discriminator=False)

#######################
###### Callbacks ######
#######################

ckpt_callback = dict(every_n_train_steps=4000, save_top_k=1, save_last=True, monitor='fid_test', mode='min')
synthesize_callback_train = dict(num_samples=12, eval_every=2000) # TODO: 4000
synthesize_callback_test = dict(num_samples=6, eval_every=2000)
fid_callback = dict(eval_every=4000)
