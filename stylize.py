import torch
from torchvision import transforms
from PIL import Image
import os
from models import Generator  # Ensure this matches your Generator class definition

def stylize_image(content_image_path, output_image_path, generator_weights_path, device):
    # Load the trained generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_weights_path, map_location=device))
    generator.eval()

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size if necessary
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training normalization
    ])

    # Load and preprocess the content image
    content_image = Image.open(content_image_path).convert('RGB')
    input_tensor = transform(content_image).unsqueeze(0).to(device)  # Add batch dimension

    # Generate the stylized image
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Post-process the output tensor to convert it back to an image
    output_tensor = output_tensor.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    output_tensor = (output_tensor * 0.5) + 0.5  # Denormalize to [0, 1]
    output_tensor = torch.clamp(output_tensor, 0, 1)  # Ensure values are within [0, 1]
    output_image = transforms.ToPILImage()(output_tensor)

    # Save the stylized image
    output_image.save(output_image_path)
    print(f"Stylized image saved to {output_image_path}")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stylize_image(
        content_image_path=,
        output_image_path=args.output_image,
        generator_weights_path="best_generator.pth",
        device=device
    )
