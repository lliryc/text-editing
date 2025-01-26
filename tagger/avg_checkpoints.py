import torch
import os


def average_checkpoints(checkpoint_dir, output_path):
    """
    Average all pytorch_model.bin checkpoints in the specified directory's subdirectories.

    Args:
        checkpoint_dir (str): Path to the directory containing checkpoint subdirectories.
        output_path (str): Path to save the averaged checkpoint.

    Returns:
        None
    """
    # Get all subdirectories containing pytorch_model.bin
    checkpoint_files = [
        os.path.join(root, "pytorch_model.bin")
        for root, _, files in os.walk(checkpoint_dir)
        if "pytorch_model.bin" in files
    ]
    
    print(f"Found {len(checkpoint_files)} checkpoints to average.")
    if not checkpoint_files:
        raise ValueError("No pytorch_model.bin files found in the specified directory.")
    
    # Initialize a dictionary to hold the averaged weights
    averaged_state_dict = None

    for i, checkpoint_path in enumerate(checkpoint_files):
        print(f"Loading checkpoint {i+1}/{len(checkpoint_files)}: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Initialize averaged_state_dict on the first iteration
        if averaged_state_dict is None:
            averaged_state_dict = state_dict
            for key in averaged_state_dict.keys():
                averaged_state_dict[key] = averaged_state_dict[key].clone()  # Ensure no reference sharing
        else:
            for key in state_dict.keys():
                averaged_state_dict[key] += state_dict[key]

    # Divide by the number of checkpoints to get the average
    num_checkpoints = len(checkpoint_files)
    for key in averaged_state_dict.keys():
        if averaged_state_dict[key].dtype in [torch.float32, torch.float64]:
            averaged_state_dict[key] /= num_checkpoints
        elif averaged_state_dict[key].dtype == torch.int64:  # Handle LongTensor
            averaged_state_dict[key] = (averaged_state_dict[key].float() / num_checkpoints).long()

    # Save the averaged checkpoint
    torch.save(averaged_state_dict, output_path)
    print(f"Averaged checkpoint saved to {output_path}")



# Example Usage
checkpoint_dir = "/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_20/qalb14"
output_path = "/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_20/qalb14/checkpoint-avg/pytorch_model.bin"
average_checkpoints(checkpoint_dir, output_path)

