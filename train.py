import os
import matplotlib.pyplot as plt
# from diffusion_model import DiffusionModel
from diffusion_model_t_cond import DiffusionModel
from diffusion_model import visualize_direction_across_latents

import torch
# — Configuration —
model_name  = "google/ddpm-ema-celebahq-256"
sample_dim  = (1, 3, 256, 256)
save_root   = "/home/ubuntu/unsupervised_dir/result_cirriculumn"
os.makedirs(save_root, exist_ok=True)

# — Build model —
dm = DiffusionModel(
    sample_dim=sample_dim,
    model_name=model_name,
    num_directions=16,
    target_timestep=800,
    duration_of_change=100,
    num_inference_steps=100,
    use_resnet=True,
)

# — Loss history buffers —
loss_hist = {
    "total": [],
    "ce":    [],
    "shift": [],
    "lpips": []
}
initial_max_mag = 0.7    # start very small
final_max_mag   = 1.5    # same as your old constant
num_epochs      = 200    # total training epochs
# — Training loop —
for epoch in range(num_epochs):
    frac_done   = epoch / (num_epochs - 1)
    curr_max_mag = initial_max_mag * ((final_max_mag/initial_max_mag) ** frac_done)
    edits, origs, dirs, L, Lc, Ls, Ll = dm.train_step(
        batch_size=2, M=16, magnitude=curr_max_mag
    )
    print(f"[Epoch {epoch:03d}] total={L:.3f}, ce={Lc:.3f}, shift={Ls:.3f}, lpips={Ll:.3f}")

    # 1) Record
    loss_hist["total"].append(L)
    loss_hist["ce"].append(Lc)
    loss_hist["shift"].append(Ls)
    loss_hist["lpips"].append(Ll)

    # 2) Every 5 epochs: plot all loss curves
    if (epoch + 1) % 5 == 0:
        plt.figure(figsize=(6,4))
        x = list(range(epoch+1))
        plt.plot(x, loss_hist["total"], label="Total")
        plt.plot(x, loss_hist["ce"],    label="CE")
        plt.plot(x, loss_hist["shift"], label="L1 shift")
        plt.plot(x, loss_hist["lpips"], label="LPIPS")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss curves up to epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(save_root, f"loss_epoch_{epoch:03d}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    # 3) Always save the per‐direction before/after grids
    dm.visualize_edirections(
        edits, origs, dirs,
        save_dir=f"{save_root}/dirs_step_{epoch}.png",
    )
    if epoch % 25 == 0:
        ckpt = {
                # models
                "aux_net": dm.aux_net.state_dict(),
                "regressor": dm.regressor.state_dict(),
                "opt_aux": dm.opt_aux.state_dict(),
                "opt_reg": dm.opt_reg.state_dict(),
            }
        torch.save(ckpt, f"{save_root}/diffusion_full_ckpt_{epoch}.pt")
    # 4) Every 100 epochs: save the latent‐interp grid for direction 3
    if epoch % 100 == 0 and epoch >0:
        visualize_direction_across_latents(
            dm,
            direction_idx=3,
            steps=5,
            max_mag=2.0,
            seed=42,
            save_dir=os.path.join(
                save_root,
                f"dir3_interpolations_epoch_{epoch:03d}.png"
            )
        )
ckpt = {
                # models
                "aux_net": dm.aux_net.state_dict(),
                "regressor": dm.regressor.state_dict(),
                "opt_aux": dm.opt_aux.state_dict(),
                "opt_reg": dm.opt_reg.state_dict(),
            }
torch.save(ckpt, f"{save_root}/diffusion_full_ckpt_{epoch}.pt")