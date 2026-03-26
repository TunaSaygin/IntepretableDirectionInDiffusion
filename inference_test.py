import os
import argparse
import torch
# from diffusion_model import DiffusionModel
from diffusion_model_t_cond import DiffusionModel
from diffusion_model import visualize_direction_across_latents


def main():
    parser = argparse.ArgumentParser(
        description="Visualize direction interpolations across all edit directions."
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="Pretrained diffusion model name or path (e.g., 'google/ddpm-ema-celebahq-256')."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the saved model checkpoint (.pt file containing 'aux_net' and 'regressor' state dicts)."
    )
    parser.add_argument(
        "--save-root", type=str, required=True,
        help="Directory where the visualization images will be saved."
    )
    parser.add_argument(
        "--epoch", type=int, required=True,
        help="Epoch number to include in the output filenames."
    )
    parser.add_argument(
        "--num-directions", type=int, required=True,
        help="Number of edit directions the model was trained with."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for visualization (default: 1)."
    )
    parser.add_argument(
        "--steps", type=int, default=5,
        help="Number of interpolation steps (default: 5)."
    )
    parser.add_argument(
        "--max-mag", type=float, default=2.0,
        help="Maximum magnitude for the direction shift (default: 2.0)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (e.g., 'cuda' or 'cpu'). If not set, will choose cuda if available."
    )
    args = parser.parse_args()

    # Choose device
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Prepare save directory
    os.makedirs(args.save_root, exist_ok=True)

    # Instantiate the diffusion model (with specified num_directions)
    dm = DiffusionModel(
        sample_dim=(args.batch_size, 3, 256, 256),
        model_name=args.model_name,
        num_directions=args.num_directions,
        target_timestep=800,
        duration_of_change=100,
        num_inference_steps=50,
        device=args.device,
        use_resnet=True,
    )

    # Load checkpoint dict
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Expecting keys 'aux_net' and 'regressor'
    if isinstance(ckpt, dict) and 'aux_net' in ckpt:
        dm.aux_net.load_state_dict(ckpt['aux_net'])
        dm.regressor.load_state_dict(ckpt['regressor'])
        print("Loaded aux_net and regressor weights from checkpoint.")
    else:
        raise ValueError("Checkpoint file must contain 'aux_net' and 'regressor' state dicts.")

    # Move model to device and eval
    # dm.eval()

    # Visualize each direction
    for dir_idx in range(args.num_directions):
        save_path = os.path.join(
            args.save_root,
            f"dir{dir_idx}_interpolations_epoch_{args.epoch:03d}.png"
        )
        print(f"Visualizing direction {dir_idx}/{args.num_directions-1}: saving to {save_path}")
        visualize_direction_across_latents(
            dm,
            direction_idx=dir_idx,
            steps=args.steps,
            max_mag=args.max_mag,
            seed=args.seed,
            save_dir=save_path,
        )


if __name__ == "__main__":
    main()
