import argparse
import torch
from pytorch_lightning import LightningModule

def filter_checkpoint(checkpoint_path, minimal_checkpoint_path):
    """
    Load a PyTorch Lightning checkpoint, remove keys containing a specified substring,
    and save the filtered state dictionary to a new checkpoint file.

    Args:
    - checkpoint_path (str): Path to the PyTorch Lightning checkpoint.
    - key_substring (str): Substring to filter out keys from the state dictionary.
    - minimal_checkpoint_path (str): Path to save the minimal checkpoint.
    """
    # Load the PyTorch Lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Extract the model's state dictionary
    model_state_dict = checkpoint['state_dict']
    key_substring = 'text_encoder.'
    # Remove keys that contain the specified substring
    keys_to_remove = [key for key in model_state_dict.keys() if key_substring in key]
    for key in keys_to_remove:
        del model_state_dict[key]

    model_state_dict = {k.replace('denoiser.', ''): v for k, v in model_state_dict.items()}


    # Save the minimal checkpoint
    torch.save(model_state_dict, f'{minimal_checkpoint_path}/min_checkpoint.ckpt')

    print(f"Minimal checkpoint saved to {minimal_checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and save a minimal PyTorch Lightning checkpoint.")
    parser.add_argument('checkpoint_path', type=str, help="Path to the PyTorch Lightning checkpoint.")
    parser.add_argument('output_dir', type=str, help="Directory to save the minimal checkpoint.")
    
    args = parser.parse_args()
    
    filter_checkpoint(args.checkpoint_path, args.output_dir)
