
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StyleTransferDataset
from models import Generator, Discriminator, FeatureExtractor


def gram_matrix(features):
    b, ch, h, w = features.size()
    features = features.view(b, ch, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
    # gram = gram / (ch * h * w)
    return gram

def compute_content_loss(gen_features, content_features):
    # Use the deepest layer for content loss
    content_loss = F.mse_loss(gen_features, content_features)
    return content_loss

# Compute style loss
def compute_style_loss(gen_features, style_features):
    style_loss = 0
    for i in range(len(gen_features)):
        gm_gf = gram_matrix(gen_features[i])
        gm_sf = gram_matrix(style_features[i])
        
        style_loss += F.mse_loss(gm_gf, gm_sf)
    return style_loss

# Compute total loss
def compute_total_loss(content_loss, style_loss, adversarial_loss, alpha, beta, gamma):


    # Compute total loss with adjusted weights
    total_loss = alpha * content_loss + beta * style_loss + gamma * adversarial_loss
    return total_loss

# def normalize_tensor(

# Prepare DataLoaders
def prepare_dataloaders(base_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    dataloaders = {}
    for subset in ["train", "val", "test"]:
        dataset = StyleTransferDataset(
            content_dir=os.path.join(base_dir, subset, 'content'),
            style_dir=os.path.join(base_dir, subset, 'style'),
            transform=transform
        )
        dataloaders[subset] = DataLoader(dataset, batch_size=batch_size, shuffle=(subset == "train"))
    return dataloaders

# Training Loop
def train(generator, discriminator, feature_extractor, dataloaders, epochs, alpha, beta, gamma, device):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999), weight_decay=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-6, betas=(0.5, 0.999), weight_decay=1e-4)
    adversarial_loss_fn = nn.BCELoss()

    # Initialize lists to store losses per iteration
    g_iteration_losses = []
    d_iteration_losses = []
    content_iteration_losses = []
    style_iteration_losses = []
    adversarial_iteration_losses = []

    g_train_losses = []
    d_train_losses = []
    g_val_losses = []
    content_val_losses = []
    style_val_losses = []

    min_val_loss = float('inf')
    iteration = 0  # Global iteration counter

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0

        # Training Phase
        for content_images, style_images in dataloaders["train"]:
            content_images, style_images = content_images.to(device), style_images.to(device)

            # Train Discriminator
            generated_images = generator(content_images).detach()
            real_inputs = torch.cat((content_images, style_images), dim=1)
            fake_inputs = torch.cat((content_images, generated_images), dim=1)

            real_inputs += torch.randn_like(real_inputs) * 0.05
            fake_inputs += torch.randn_like(fake_inputs) * 0.05

            real_validity = discriminator(real_inputs)
            fake_validity = discriminator(fake_inputs)

            real_labels = torch.ones_like(real_validity, device=device) * 0.9
            fake_labels = torch.zeros_like(fake_validity, device=device) + 0.1

            real_loss = adversarial_loss_fn(real_validity, real_labels)
            fake_loss = adversarial_loss_fn(fake_validity, fake_labels)
            d_loss = (real_loss + fake_loss) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            generated_images = generator(content_images)
            fake_inputs = torch.cat((content_images, generated_images), dim=1)
            fake_validity = discriminator(fake_inputs)
            adv_loss = adversarial_loss_fn(fake_validity, real_labels)  # Generator tries to fool discriminator


            style_features_gen, content_feature_gen = feature_extractor(generated_images)
            style_features_style, content_feature_style = feature_extractor(style_images)
            style_features_content, content_feature_content = feature_extractor(content_images)


            content_loss = compute_content_loss(content_feature_gen, content_feature_content)
            style_loss = compute_style_loss(style_features_gen, style_features_style)


            g_loss = compute_total_loss(content_loss, style_loss, adv_loss, alpha, beta, gamma)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            # Append iteration-level losses
            g_iteration_losses.append(g_loss.item())
            d_iteration_losses.append(d_loss.item())
            content_iteration_losses.append(content_loss.item())
            style_iteration_losses.append(style_loss.item())
            adversarial_iteration_losses.append(adv_loss.item())

            iteration += 1  # Increment the iteration counter

        # Calculate average losses over the epoch
        avg_g_loss = epoch_g_loss / len(dataloaders["train"])
        avg_d_loss = epoch_d_loss / len(dataloaders["train"])
        g_train_losses.append(avg_g_loss)
        d_train_losses.append(avg_d_loss)

        # Validation Phase
        generator.eval()
        val_g_loss = 0
        val_content_loss = 0
        val_style_loss = 0

        with torch.no_grad():
            for content_images, style_images in dataloaders["val"]:
                content_images, style_images = content_images.to(device), style_images.to(device)
                generated_images = generator(content_images)

                style_features_gen, content_feature_gen = feature_extractor(generated_images)
                style_features_style, content_feature_style = feature_extractor(style_images)
                style_features_content, content_feature_content = feature_extractor(content_images)


                content_loss = compute_content_loss(content_feature_gen, content_feature_content)
                style_loss = compute_style_loss(style_features_gen, style_features_style)

                g_loss= compute_total_loss(content_loss, style_loss, 0, alpha, beta, 0)  # No adversarial loss during validation

                val_g_loss += g_loss.item()
                val_content_loss += content_loss.item() * alpha
                val_style_loss += style_loss.item() * beta

        # Calculate average validation losses
        avg_val_g_loss = val_g_loss / len(dataloaders["val"])
        avg_val_content_loss = val_content_loss / len(dataloaders["val"])
        avg_val_style_loss = val_style_loss / len(dataloaders["val"])

        g_val_losses.append(avg_val_g_loss)
        content_val_losses.append(avg_val_content_loss)
        style_val_losses.append(avg_val_style_loss)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Train - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        print(f"Val   - G Loss: {avg_val_g_loss:.4f}, Content Loss: {avg_val_content_loss:.4f}, Style Loss: {avg_val_style_loss:.4f}")

        # os.makedirs("generated_training", exist_ok=True)



        # Save the model with the lowest validation loss
        if avg_val_g_loss < min_val_loss:
            min_val_loss = avg_val_g_loss
            torch.save(generator.state_dict(), "best_generator_2.pth")
            torch.save(discriminator.state_dict(), "best_discriminator.pth")
            print("Best model saved based on validation loss.")

    # Plot iteration-level losses after training
    # Plot generator and discriminator losses
    plt.figure(figsize=(12, 6))
    plt.plot(g_iteration_losses, label="Generator Loss")
    plt.plot(d_iteration_losses, label="Discriminator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss per Iteration")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_per_iteration.png")
    plt.show()

    # Plot content, style, and adversarial losses
    plt.figure(figsize=(12, 6))
    plt.plot(content_iteration_losses, label="Content Loss")
    plt.plot(style_iteration_losses, label="Style Loss")
    plt.plot(adversarial_iteration_losses, label="Adversarial Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Components per Iteration")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_components_per_iteration.png")
    plt.show()

    # Plot epoch-level generator loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), g_train_losses, label="Generator Train Loss")
    plt.plot(range(1, epochs + 1), g_val_losses, label="Generator Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("generator_epoch_loss_curve.png")
    plt.show()

    # Plot epoch-level discriminator loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), d_train_losses, label="Discriminator Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("discriminator_epoch_loss_curve.png")
    plt.show()

    # Save final models
    torch.save(generator.state_dict(), "final_generator_2.pth")
    torch.save(discriminator.state_dict(), "final_discriminator_2.pth")
    print("Final models saved.")


def test(generator, dataloader, device):
    generator.eval()
    image_count = 0
    with torch.no_grad():
        for idx, (content_images, _) in enumerate(dataloader):
            content_images = content_images.to(device)
            generated_images = generator(content_images)

            # Denormalize and convert to CPU
            content_images = (content_images.cpu() * 0.5) + 0.5
            generated_images = (generated_images.cpu() * 0.5) + 0.5

            # Save images
            for i in range(len(content_images)):
                content_img = transforms.ToPILImage()(content_images[i])
                generated_img = transforms.ToPILImage()(generated_images[i])

                content_img.save(f"test_results_2/content_{idx}_{i}.png")
                generated_img.save(f"test_results_2/generated_{idx}_{i}.png")
                image_count += 1

    print(f"Total images saved: {image_count}")

# Main Script
if __name__ == "__main__":

    # Prepare dataloaders
    dataloaders = prepare_dataloaders("datasets", batch_size=64)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    feature_extractor = FeatureExtractor().to(device)
    generator.load_state_dict(torch.load("best_generator_2.pth"))
    # Train the model
    train(generator, discriminator, feature_extractor, dataloaders, epochs=300, alpha=0.7, beta=1e-6, gamma=1, device=device)

    # Load the best model for testing
    generator.load_state_dict(torch.load("best_generator_2.pth"))

    # Create directory for test results
    os.makedirs("test_results_2", exist_ok=True)

    # Test the model
    test(generator, dataloaders["test"], device=device)


