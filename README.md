# Distributed_DL_training
Trained a custom ResNet 18 model and trained on 4 T4 gpus on GCP instance using distributed DL module from PyTorch

# Run the following command to train the model on distributed setup
torchrun --nproc_per_node=<NUMBER OF GPUS> train.py --batch_size=<BATCH SIZE> --epochs=<EPOCHS>
