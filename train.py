
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StyleTransferDataset
from models import Generator, Discriminator, FeatureExtractor, LossTracker


def gram_matrix(features):
    b, ch, h, w = features.size()
    features = features.view(b, ch, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
    gram = gram / (ch * h * w)
    return gram

def compute_content_loss(gen_features, content_features):
    # Use the deepest layer for content loss
    content_loss = F.mse_loss(gen_features[4], content_features[4])
    return content_loss

# Compute style loss
def compute_style_loss(gen_features, style_features):
    style_loss = 0
    # You can choose which layers to include in style loss
    style_layers = [0, 1, 2, 3]  # Exclude the deepest layer (content layer)
    for i in style_layers:
        gm_gf = gram_matrix(gen_features[i])
        gm_sf = gram_matrix(style_features[i])
        
        style_loss += F.mse_loss(gm_gf, gm_sf)
    return style_loss

# Compute total loss
def compute_total_loss(content_loss, style_loss, adversarial_loss, alpha, beta, gamma, loss_tracker):
    # Get running averages of losses
    content_mean, style_mean, adversarial_mean = loss_tracker.get_means()
    epsilon = 1e-8

    # Compute scaling factors inversely proportional to the loss magnitudes
    content_scale = 1.0 / (content_mean + epsilon)
    style_scale = 1.0 / (style_mean + epsilon)
    adversarial_scale = 1.0 / (adversarial_mean + epsilon)

    # Normalize the losses
    normalized_content_loss = content_loss * content_scale
    normalized_style_loss = style_loss * style_scale
    normalized_adversarial_loss = adversarial_loss * adversarial_scale

    # Compute total loss with adjusted weights
    total_loss = alpha * normalized_content_loss + beta * normalized_style_loss + gamma * normalized_adversarial_loss
    return total_loss


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
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    adversarial_loss_fn = nn.BCELoss()

    loss_tracker = LossTracker()

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

            real_validity = discriminator(real_inputs)
            fake_validity = discriminator(fake_inputs)

            real_labels = torch.ones_like(real_validity, device=device)
            fake_labels = torch.zeros_like(fake_validity, device=device)

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

            gen_features = feature_extractor(generated_images)
            content_features = feature_extractor(content_images)
            style_features = feature_extractor(style_images)
            content_loss = compute_content_loss(gen_features, content_features)
            style_loss = compute_style_loss(gen_features, style_features)

            loss_tracker.update(content_loss, style_loss, adv_loss)

            g_loss = compute_total_loss(content_loss, style_loss, adv_loss, alpha, beta, gamma, loss_tracker)

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

                gen_features = feature_extractor(generated_images)
                content_features = feature_extractor(content_images)
                style_features = feature_extractor(style_images)

                content_loss = compute_content_loss(gen_features, content_features)
                style_loss = compute_style_loss(gen_features, style_features)

                g_loss = compute_total_loss(content_loss, style_loss, 0, alpha, beta, 0, loss_tracker)  # No adversarial loss during validation

                val_g_loss += g_loss.item()
                val_content_loss += content_loss.item()
                val_style_loss += style_loss.item()

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

        # Save the model with the lowest validation loss
        if avg_val_g_loss < min_val_loss:
            min_val_loss = avg_val_g_loss
            torch.save(generator.state_dict(), "best_generator.pth")
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
    torch.save(generator.state_dict(), "final_generator.pth")
    torch.save(discriminator.state_dict(), "final_discriminator.pth")
    print("Final models saved.")


def test(generator, dataloader, device):
    generator.eval()
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

                content_img.save(f"test_results/content_{idx}_{i}.png")
                generated_img.save(f"test_results/generated_{idx}_{i}.png")

# Main Script
if __name__ == "__main__":

    # Prepare dataloaders
    dataloaders = prepare_dataloaders("datasets", batch_size=64)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    feature_extractor = FeatureExtractor().to(device)
    # Train the model
    train(generator, discriminator, feature_extractor, dataloaders, epochs=30, alpha=1, beta=1, gamma=5, device=device)

    # Load the best model for testing
    generator.load_state_dict(torch.load("best_generator.pth"))

    # Create directory for test results
    os.makedirs("test_results", exist_ok=True)

    # Test the model
    test(generator, dataloaders["test"], device=device)

