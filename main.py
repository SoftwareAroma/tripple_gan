import os
import torch
from torch.utils.data import DataLoader
from src.networks import Generator, Discriminator, Classifier
from src.model import TripleGAN
from src.utils import load_dataset, save_images, get_last_checkpoint
from options import parse_args

# Function to train or test the model
def train_or_test(options):
    print("Loading dataset...")
    dataset = load_dataset(options.root_dir, options.image_size)
    print(f"Dataset loaded with {len(dataset)} images.")
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    generator = Generator(input_dim=options.latent_dim)
    discriminator = Discriminator()
    classifier = Classifier(num_classes=options.num_classes)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=options.lr_G, betas=options.beta_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=options.lr_D, betas=options.beta_D)
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=options.lr_C, betas=options.beta_C)

    gan_model = TripleGAN(
        generator=generator,
        discriminator=discriminator,
        classifier=classifier,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        optimizer_C=optimizer_C,
        options=options,
        latent_dim=options.latent_dim
    )    

    if options.train:
        print("Starting training...")
        for epoch in range(options.num_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                d_loss, g_loss, c_loss = gan_model.train_step(imgs, labels)

            print(f"Epoch [{epoch+1}/{options.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}")
            # create the checkpoint directory if it does not exist
            if not os.path.exists(options.checkpoint_dir):
                os.makedirs(options.checkpoint_dir)
            checkpoint_path = os.path.join(options.checkpoint_dir, f'checkpoint_{epoch+1}.pth')
            gan_model.save_checkpoint(checkpoint_path)
        
        # Save the model after training
        if not os.path.exists(options.output_path):
            os.makedirs(options.output_path)
        gan_model.save_model(os.path.join(options.output_path, options.model_path))
    elif options.continue_training:
        print("Continuing training from the last checkpoint...")
        # get the last checkpoint in the checkpoint directory
        last_checkpoint, last_epoch = get_last_checkpoint(options.checkpoint_dir)
        if os.path.exists(last_checkpoint):
            print(f"Loading checkpoint from {last_checkpoint}")
            gan_model.load_model(last_checkpoint)
            # Continue training from the last epoch
            for epoch in range(last_epoch, options.num_epochs):
                for i, (imgs, labels) in enumerate(dataloader):
                    d_loss, g_loss, c_loss = gan_model.train_step(imgs, labels)

                print(f"Epoch [{epoch+1}/{options.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}")
                checkpoint_path = os.path.join(options.checkpoint_dir, f'checkpoint_{epoch+1}.pth')
                gan_model.save_checkpoint(checkpoint_path)
        else:
            print(f"No checkpoint found at {last_checkpoint}, starting from scratch.")
    else:
        print("Testing the model...")
        checkpoint_path = os.path.join(options.output_path, options.model_path)
        # Load the model for testing if not training
        print(f"Loading model from {checkpoint_path} for testing.")
        gan_model.load_model(checkpoint_path)
        n_images = options.n_images
        images = gan_model.generate_images(n_images=n_images)
        save_images(
            images=images,
            output_path=os.path.join(options.output_path, 'images'),
            n_images=n_images
        )

if __name__ == "__main__":
    options = parse_args()
    train_or_test(options)
