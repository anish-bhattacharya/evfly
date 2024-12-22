import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Load a PyTorch .pth file and optionally rename a state_dict key.")
    parser.add_argument("--pth_file", type=str, help="Path to the .pth file to load.")
    parser.add_argument("--old_key", type=str, default=None, help="The existing key name in the state_dict to rename.")
    parser.add_argument("--new_key", type=str, default=None, help="The new key name to assign to the given old_key.")
    parser.add_argument("--save", type=str, default=None, help="If provided, the modified dictionary will be saved to this file.")
    
    args = parser.parse_args()

    # Load the .pth file
    checkpoint = torch.load(args.pth_file, map_location="cpu")

    # The checkpoint might contain a 'state_dict' key if it's a model checkpoint.
    # If not, we'll assume the loaded object itself is the state dict.
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        # In case the loaded file is already a state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError("Loaded .pth file does not contain a recognizable dict or 'state_dict'.")

    # Print the entire state_dict keys and values
    print("Current state_dict keys and values:")
    for k, v in state_dict.items():
        print(f"{k}: {type(v)} - {v.shape if hasattr(v, 'shape') else ''}")

    # If renaming is requested
    if args.old_key is not None and args.new_key is not None:
        if args.old_key in state_dict:
            # Rename the key
            state_dict[args.new_key] = state_dict.pop(args.old_key)
            print(f"\nRenamed key '{args.old_key}' to '{args.new_key}'.")

            # If we're modifying the original checkpoint structure, do that:
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint["state_dict"] = state_dict
            else:
                checkpoint = state_dict

            # If the user wants to save the modified checkpoint
            if args.save is not None:
                torch.save(checkpoint, args.save)
                print(f"Modified checkpoint saved to {args.save}")
        else:
            print(f"Key '{args.old_key}' not found in state_dict.")
    else:
        # If no rename requested but a save path is provided, we still allow saving the unchanged checkpoint
        if args.save is not None:
            torch.save(checkpoint, args.save)
            print(f"Unmodified checkpoint saved to {args.save}")

if __name__ == "__main__":
    main()
