set -ex
python main.py --root_dir datasets/fundusimage1000 --continue_training
# python main.py --root_dir "/path/to/dataset" --train --image_size 128 --batch_size 64 --num_epochs 100 --lr_G 0.0002 --lr_D 0.0002 --lr_C 0.0002 --latent_dim 100
