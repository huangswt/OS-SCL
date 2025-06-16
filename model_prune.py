import torch
import argparse
import os


def prune_model(original_checkpoint_path, remove_prefixes, overwrite=False):
    if not overwrite:
        dir_path = os.path.dirname(original_checkpoint_path)
        new_checkpoint_path = os.path.join(dir_path, "pruned_model.pth")
    else:
        new_checkpoint_path = original_checkpoint_path

    state_dict = torch.load(original_checkpoint_path, map_location='cpu')

    new_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in remove_prefixes)
    }

    torch.save(new_state_dict, new_checkpoint_path)
    print(f"Pruned model weights saved to {new_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune specific layers from a PyTorch model checkpoint.")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to the original model checkpoint")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite the original model file if set")

    args = parser.parse_args()

    remove_prefixes = [
        'TFgramNet.pre_block4',
        'TFgramNet.spec_augmenter',
        'TFgramNet.bn0',
        'TFgramNet.conv_block1',
        'TFgramNet.conv_block2',
        'TFgramNet.conv_block3',
        'TFgramNet.conv_block4',
        'TFgramNet.conv_block5',
        'TFgramNet.conv_block6',
        'TFgramNet.fc1',
        'TFgramNet.fc_audioset'
    ]

    prune_model(args.input, remove_prefixes, args.overwrite)
