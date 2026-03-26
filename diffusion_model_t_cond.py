from diffusers import UNet2DModel, DDIMScheduler
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, ResNet18_Weights
from enum import Enum
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import torch.nn.functional as F
import lpips

class DirectionRegressor(nn.Module):
    def __init__(self, in_channels, image_dim, num_directions, width=2):
        super(DirectionRegressor, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels * 2, 3 * width, kernel_size=5),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=5),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=5),
            nn.BatchNorm2d(60 * width),
            nn.ReLU(),
        )

        # Dummy input to find the shape AFTER global avg pooling
        dummy_input = torch.randn(1, in_channels * 2, image_dim[0], image_dim[1])
        dummy_out = self.convnet(dummy_input)
        dummy_out = dummy_out.mean(dim=[2, 3])  # <--- same pooling as in forward
        flatten_size = dummy_out.shape[1]

        self.fc_logits = nn.Sequential(
            nn.Linear(flatten_size, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, num_directions),
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(flatten_size, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1),
        )

    def forward(self, original, modified):
        x = torch.cat([original, modified], dim=1)
        features = self.convnet(x)
        features = features.mean(dim=[2, 3])  # same as dummy pass
        logits = self.fc_logits(features)
        shift = self.fc_shift(features).squeeze()
        return logits, shift


def save_hook(module, input, output):
    setattr(module, "output", output)


class ResnetRegressor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(ResnetRegressor, self).__init__()
        self.features_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        nn.init.kaiming_normal_(
            self.features_extractor.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.prod(dim))
        self.shift_estimator = nn.Sequential(
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512, 1))

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(
                x2, self.downsample
            )
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


class DeformatorType(
    Enum
):  # I don't know why it is like that but original implementation put it so I am putting it like that
    FC = 1
    LINEAR = 2
    ID = 3
    ORTHO = 4
    PROJECTIVE = 5
    RANDOM = 6


class AuxiliaryNetwork(nn.Module):
    """
    LatentDeformator generates a shift (delta_h) for the bottleneck features.
    Instead of using a factorized A matrix and spatial masks, this version
    maps a one-hot (or few-hot) direction vector through a fully-connected network
    to produce a shift tensor with the same spatial shape as the bottleneck.
    """

    def __init__(
        self,
        bottleneck_channels,
        num_directions,
        num_edit_steps,
        bottleneck_spatial_dim=(8, 8),
        inner_dim=1024,
        type=DeformatorType.FC,
        random_init=False,
        bias=True,
    ):
        super(AuxiliaryNetwork, self).__init__()
        self.num_directions = num_directions
        self.num_edit_steps = num_edit_steps
        self.bottleneck_channels = bottleneck_channels
        self.bottleneck_spatial_dim = tuple(bottleneck_spatial_dim)
        # The shift (or delta) should match the bottleneck shape: (C, H, W)
        self.shift_dim = (
            self.num_edit_steps,
            self.bottleneck_channels,
            self.bottleneck_spatial_dim[0],
            self.bottleneck_spatial_dim[1],
        )
        # For a one-hot/few-hot vector, we take the input dimension to be num_directions.
        self.input_dim = num_directions
        # The output dimension is the flattened shift (i.e. product of shift_dim)
        self.out_dim = int(np.prod(self.shift_dim))
        self.type = type

        if self.type == DeformatorType.FC:
            self.fc1 = nn.Linear(self.input_dim, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim, inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)
        else:
            raise NotImplementedError(
                "Only FC deformator is implemented in this example."
            )

    def forward(
        self,
        batch_size,
        device,
        direction_idx=None,
        magnitude=1.0,
        random_directions=False,
        binary_vectors=None,
    ):
        """
        Generate a shift based on a provided direction vector and apply it to the bottleneck.

        Args:
            bottleneck (torch.Tensor): UNet bottleneck features of shape (B, C, H, W)
            direction_idx (int or None): If provided, use this direction for all samples.
            magnitude (float): Scalar to control the strength of the shift.
            random_directions (bool): If True, randomly select 1-3 active directions per sample.
            binary_vectors (torch.Tensor or None): Pre-generated binary direction vectors.

        Returns:
            Modified bottleneck features of shape (B, C, H, W).
        """
        # batch_size = bottleneck.shape[0]
        # device = bottleneck.device

        # Create the direction vector input (one-hot or few-hot) for each sample
        if binary_vectors is not None:
            direction_vectors = binary_vectors  # Expected shape: (B, num_directions)
        elif random_directions:
            direction_vectors = torch.zeros(
                batch_size, self.num_directions, device=device
            )
            for i in range(batch_size):
                num_active = torch.randint(
                    1, min(4, self.num_directions + 1), (1,)
                ).item()
                active_indices = torch.randperm(self.num_directions)[:num_active]
                direction_vectors[i, active_indices] = 1.0
        elif direction_idx is not None:
            direction_vectors = torch.zeros(
                batch_size, self.num_directions, device=device
            )
            direction_vectors[torch.arange(batch_size), direction_idx] = 1.0
            # print(direction_vectors)
        else:
            print(f"Entered no change area!!!")
            raise Exception
            # If no direction is specified, no change is applied.
            # return bottleneck
            # return 0

        # Pass the direction vector through the FC network to produce a shift
        x = direction_vectors.view(batch_size, self.input_dim)
        if self.type == DeformatorType.FC:
            x1 = self.fc1(x)
            x = self.act1(self.bn1(x1))
            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))
            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))
            shift_flat = self.fc4(x)
        else:
            print("Big problem in aux network!!!")
            shift_flat = x  # This branch won't be reached in the current implementation
        # print(f"Shift flat")
        # print(shift_flat)

        # Ensure the flat shift has the correct number of elements
        flat_shift_dim = self.out_dim
        if shift_flat.shape[1] < flat_shift_dim:
            print(f"Problem in shift_flat matrix adding padding...")
            padding = torch.zeros(
                [batch_size, flat_shift_dim - shift_flat.shape[1]],
                device=shift_flat.device,
            )
            shift_flat = torch.cat([shift_flat, padding], dim=1)
        elif shift_flat.shape[1] > flat_shift_dim:
            print(f"Problem in shift_flat matrix truncating...")
            shift_flat = shift_flat[:, :flat_shift_dim]

        try:
            shift = shift_flat.view(batch_size, *self.shift_dim)
        except Exception:
            shift = shift_flat

        # Scale the shift by the provided magnitude
        print(f"magnitude raw dim {magnitude.shape}")
        if torch.is_tensor(magnitude):
            # reshape to (B,1,1,1) so it multiplies each sample separately
            magnitude = magnitude.view(batch_size, 1, 1, 1, 1)
            # magnitude = magnitude.unsqueeze(0)

        print(f"shift dim {shift.shape}")
        print(f"magnitude dim {magnitude.shape}")
        shift = shift * magnitude # (10 x 16 x 512 x 8 x 8 ) x (16)
        # print("-"*20+"\n"+""*10+"Actual Shift"+"\n"+"-"*20)
        # print(shift)
        # Add the computed shift to the original bottleneck
        # (Assumes bottleneck shape (B, C, H, W) matches shift shape)
        # print(f"Shift shape: {shift.shape}")
        # print(f"Bottleneck shape: {bottleneck.shape}")

        # Add these debug prints
        # print("Direction vectors shape:", direction_vectors.shape)
        # print("Direction vectors:", direction_vectors)
        print("Shift shape:", shift.shape)
        # print("Shift mean:", shift.mean().item())
        # print("Shift std:", shift.std().item())

        return shift

    def get_delta_h(self, direction_idx, magnitude=1.0):
        """
        Get the delta_h for a specific direction (for visualization or analysis)
        """
        channel_shift = self.A[direction_idx] * self.scales[direction_idx]
        channel_shift = channel_shift.unsqueeze(-1).unsqueeze(-1)  # (C, 1, 1)
        spatial_mask = self.spatial_masks[direction_idx]  # (1, H, W)
        return channel_shift * spatial_mask * magnitude


class ModifiedUNet(UNet2DModel):

    def set_others(
        self,
        target_timestep=None,
        duration_of_change=None,
    ):
        self.target_timestep = target_timestep
        self.duration_of_change = duration_of_change
        self.current_timestep = None
        self.edited_layer_count=0


    def set_timestep(self, timestep):
        """Set the current timestep for the forward pass"""
        self.current_timestep = timestep

    def forward(self, x, timestep, edit_dirs=None, insert_aux=False):
        # Set the current timestep
        self.set_timestep(timestep)

        # Store the original mid_block forward method
        original_mid_block_forward = self.mid_block.forward

        # Define a new mid_block forward that applies the auxiliary network
        def modified_mid_block_forward(hidden_states, temb=None):
            # Get the original mid block output
            output = original_mid_block_forward(hidden_states, temb)

            if (self.current_timestep is not None
                and self.current_timestep.item() <= self.target_timestep
                and self.current_timestep.item() > self.target_timestep - self.duration_of_change
                and insert_aux):

                # print(f"output dims {output.shape}")
                # print(f"edit dirs dims {edit_dirs.shape}")
                # print(f"edited_layer_count {self.edited_layer_count}")

                print(f"t: {self.current_timestep}, in aux_inject")


                output = output + edit_dirs[:, self.edited_layer_count]

                self.edited_layer_count += 1

                return output
            else:
                # print("The mighty resetter is summoded!!!!!!!")
                output = output
                if insert_aux:
                  self.edited_layer_count = 0

            return output

        # Replace the mid_block forward method temporarily
        self.mid_block.forward = modified_mid_block_forward


        try:
            # Run the forward pass with the modified mid_block
            output = super().forward(x, timestep)
        finally:
            # Restore the original mid_block forward method
            self.mid_block.forward = original_mid_block_forward

        return output


# Custom diffusion model using a hook on the mid block.
class CustomPretrainedDiffusionModel:
    def __init__(
        self,
        model_name,
        auxiliary_net,
        target_timestep=500,
        duration_of_change=None,
        num_inference_steps=1000,
        device="cuda",
    ):
        self.device = device
        self.model_name = model_name
        self.auxiliary_net = auxiliary_net
        self.auxiliary_net.to(self.device)
        self.target_timestep = target_timestep
        self.duration_of_change = (
            duration_of_change if duration_of_change is not None else 1
        )
        # These will be updated during the denoising loop.
        self.current_timestep = None
        self.aux_inject = False  # flag to control injection
        self.aux_params = {}  # dictionary to store parameters for auxiliary network
        self.num_inference_steps = num_inference_steps

        self.load_model_and_scheduler()

    def load_model_and_scheduler(self):

        # self.unet = UNet2DModel.from_pretrained(self.model_name).to(self.device).eval()

        # Try the simpler way for ModifiedUNet
        self.unet_edited: ModifiedUNet = (
            ModifiedUNet.from_pretrained(self.model_name).to(self.device).eval()
        )

        self.unet_edited.set_others(
            target_timestep=self.target_timestep,
            duration_of_change=self.duration_of_change,
        )

        # Load the scheduler; here using DDIMScheduler as an example.
        self.scheduler = DDIMScheduler.from_pretrained(self.model_name)
        self.scheduler.alphas_cumprod.to(self.device)
        # Set inference timesteps (adjust as needed).
        self.scheduler.set_timesteps(self.num_inference_steps)

    def denoising_loop(
        self,
        init_latent,
        direction_idx=None,
        magnitude=1.0,
        random_directions=False,
        binary_vectors=None,
        inject_aux=True,
    ):
        """
        Run the iterative denoising loop. At each timestep the latent is updated.
        At the target timestep, if inject_aux is True, the auxiliary network is applied via the hook.
        """
        latent = init_latent.requires_grad_(True)
        # latent_for_edited = latent.detach().requires_grad_(True)
        # Set auxiliary parameters for the hook.
        self.aux_inject = inject_aux


        if self.aux_inject:
          edit_dirs = self.auxiliary_net(
              batch_size=latent.shape[0],
              device=self.device,
              direction_idx=direction_idx,
              magnitude=magnitude,
              random_directions=random_directions,
              binary_vectors=binary_vectors
              )

        # Iterate through timesteps provided by the scheduler (often descending).
        for t in tqdm(self.scheduler.timesteps):
            # print(f"t: {t}")
            # Ensure timestep is a float tensor on the proper device.
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device).float()
            # elif not torch.is_floating_point(t):
            #     t = t.float()
            self.current_timestep = t
            # Run the UNet forward pass. The hook on mid_block will modify its output if t equals target_timestep.
            # Note: the output from the UNet is typically a UNet2DOutput or similar. Adjust as necessary.
            # if isinstance(latent, torch.Tensor):

            # with torch.inference_mode():
            #     # noise_pred = checkpoint(lambda x, t: self.unet(x, t).sample, latent, t)
            #     noise_pred = self.unet(latent, t).sample

            # if (self.aux_inject):
                # print(f"Entered asyrp stage: timestep {t}")
                # noise_edited = checkpoint(
                #     lambda x, t, edit_dirs: self.unet_edited(x, t, edit_dirs).sample, latent, t, edit_dirs
                # )

            with torch.inference_mode():
                noise_pred = self.unet_edited(latent, t, insert_aux=False).sample

            if (self.aux_inject):
              noise_edited = checkpoint(
                      lambda x, t, edit_dirs, insert_aux: self.unet_edited(x, t, edit_dirs, insert_aux).sample, latent, t, edit_dirs, self.aux_inject
                  )

            if (self.current_timestep is not None
                and self.current_timestep.item() <= self.target_timestep
                and self.current_timestep.item() > self.target_timestep - self.duration_of_change
                and self.aux_inject):
              # print(f"noise_edited.grad_fn {noise_edited.grad_fn}")
              a_t = self.scheduler.alphas_cumprod[t]
              a_t_next = self.scheduler.alphas_cumprod[t - 1]
              p_t = (latent - torch.sqrt(1 - a_t) * noise_edited) / torch.sqrt(a_t)
              latent = (
                  torch.sqrt(a_t_next) * p_t + torch.sqrt(1 - a_t_next) * noise_pred
              )
            else:
                # Use the scheduler to update the latent.
                step_output = self.scheduler.step(noise_pred, t, latent)
                latent = step_output.prev_sample

        return latent

    def generate_image(
        self,
        init_latent,
        direction_idx=None,
        magnitude=1.0,
        random_directions=False,
        binary_vectors=None,
        inject_aux=True,
    ):
        """
        Generate a final denoised image from the initial latent.
        """
        final_latent = self.denoising_loop(
            init_latent,
            direction_idx=direction_idx,
            magnitude=magnitude,
            random_directions=random_directions,
            binary_vectors=binary_vectors,
            inject_aux=inject_aux,
        )
        return final_latent

    def generate_both_images(self, init_latent, direction_idx, magnitude=1.0):
        edited_latent = self.denoising_loop(
            init_latent,
            direction_idx=direction_idx,
            magnitude=magnitude,
            random_directions=False,
            inject_aux=True,
        )
        original_latent = self.denoising_loop(
            init_latent,
            direction_idx=direction_idx,
            magnitude=magnitude,
            random_directions=False,
            inject_aux=False,
        )
        return edited_latent, original_latent


class DiffusionModel:
    def __init__(
        self,
        sample_dim,  # e.g. (1,3,256,256)
        model_name,  # e.g. "google/ddpm-ema-celebahq-256"
        num_directions: int = 10,
        target_timestep: int = 500,
        duration_of_change: int = 1,
        num_inference_steps: int = 50,
        total_steps: int = 1000,
        device: str = None,
        use_resnet: bool = False,
        lpips_weight: float = 2,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.sample_dim = sample_dim
        self.model_name = model_name
        self.num_directions = num_directions
        self.target_timestep = target_timestep
        self.duration_of_change = duration_of_change
        self.num_inference_steps = num_inference_steps

        # 1) Build the auxiliary network
        _, C, H, W = sample_dim

        # calculate the edit steps number
        step_size = total_steps // num_inference_steps
        num_edit_steps = duration_of_change // step_size

        print(f"num_edit_steps: {num_edit_steps}")

        self.aux_net = AuxiliaryNetwork(
            num_edit_steps=num_edit_steps,
            bottleneck_channels=512,
            num_directions=num_directions,
            bottleneck_spatial_dim=(8, 8),  # match your UNet bottleneck
        ).to(self.device)

        # 2) Wrap it in your custom diffusion model
        self.custom_diffusion = CustomPretrainedDiffusionModel(
            model_name=model_name,
            auxiliary_net=self.aux_net,
            target_timestep=target_timestep,
            duration_of_change=duration_of_change,
            num_inference_steps=num_inference_steps,
            device=self.device,
        )

        # 3) Regressor: takes (orig, edited) image pairs
        # 3) Regressor: takes (orig, edited) image pairs
        if not use_resnet:
            self.regressor = DirectionRegressor(
                in_channels=3, image_dim=(H, W), num_directions=num_directions
            ).to(self.device)
        else:
            self.regressor = ResnetRegressor(dim=num_directions, downsample=None).to(
                self.device
            )
        # put models in training mode
        self.aux_net.train()
        self.regressor.train()

        # 4) Optimizers & losses
        self.opt_aux = optim.Adam(self.aux_net.parameters(), lr=1e-4)
        self.opt_reg = optim.Adam(self.regressor.parameters(), lr=1e-4)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_shift = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_weight = lpips_weight

    def train_step(self, batch_size: int, M: int, magnitude: float = 1.0):
        """
        - Draw batch_size random noises
        - For each, pick M random directions
        - Generate (edited, original) via custom_diffusion.generate_both_images(...)
        - Train regressor to predict dir index (CE loss) and shift magnitude (L1)
        - Step both aux_net and regressor
        """
        self.aux_net.train()
        self.regressor.train()

        # print("Aux Net Parameters:")
        # for name, param in self.aux_net.named_parameters():
        #     print(
        #         f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}, grad={param.grad is not None if param.requires_grad else 'N/A'}"
        #     )

        # print("\nRegressor Parameters:")
        # for name, param in self.regressor.named_parameters():
        #     print(
        #         f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}, grad={param.grad is not None if param.requires_grad else 'N/A'}"
        #     )
        # 1) Sample noise
        z = torch.randn(batch_size, *self.sample_dim[1:], device=self.device)

        # 2) Sample M directions per sample
        # dirs = torch.randint(
        #     0, self.num_directions, (batch_size, M), device=self.device
        # )

        dirs = torch.arange(M).repeat(1, batch_size).to(self.device)

        # sample a magnitude
        sampled_magnitude = random.uniform(0.4, magnitude) * random.choice([-1, 1])
        N = batch_size * M
        # uniform in [0.4, magnitude)
        mag = torch.empty(N, device=self.device).uniform_(0.4, magnitude)
        # random signs ±1
        signs = torch.randint(0, 2, (N,), device=self.device).float().mul(2).sub(1)
        sampled_magnitudes = mag * signs
        # edits, origs = [], []
        # for m in range(M):
        #     e, o = self.custom_diffusion.generate_both_images(
        #         z, direction_idx=dirs[:, m], magnitude=sampled_magnitude
        #     )
        #     edits.append(e)
        #     origs.append(o)

        # # 3) Stack → (B*M, C, H, W)
        # edits = torch.stack(edits, dim=1).view(-1, *edits[0].shape[1:])
        # origs = torch.stack(origs, dim=1).view(-1, *origs[0].shape[1:])
        # flat_dirs = dirs.view(-1)

        batch_expanded = z.repeat_interleave(M, dim=0)  # Shape: (batch_size*M, C, H, W)
        flat_dirs = dirs.view(-1)  # Shape: (batch_size*M)

        # print("batch: " + str(batch_expanded.shape))
        # print("dirs: " + str(flat_dirs.shape))

        # Generate all edits at once
        all_edits = self.custom_diffusion.generate_image(
            batch_expanded,
            direction_idx=flat_dirs,
            magnitude=sampled_magnitudes,
            inject_aux=True,
        )
        all_origs = self.custom_diffusion.generate_image(z, inject_aux=False)
        all_origs = all_origs.repeat_interleave(M, dim=0)

        edits = all_edits
        origs = all_origs

        # 4) Regressor prediction
        logits, shift_pred = self.regressor(origs, edits)

        print("logits shape", logits.shape)
        print("flat dirs shape", flat_dirs.shape)
        print("class predicted", torch.argmax(logits, dim=1))
        print("class real", flat_dirs)
        print("shift predicted", shift_pred)
        print("actual shift dim:", sampled_magnitudes)
        print("shift real", sampled_magnitudes)
        # print("logits 0", logits[0])
        # print("logits 1", logits[1])

        # 5) Compute losses
        loss_cls = self.criterion_cls(logits, flat_dirs)
        # loss_shift = self.criterion_shift(
        #     shift_pred, torch.full_like(shift_pred, sampled_magnitude)
        # )
        loss_shift = self.criterion_shift(
            shift_pred, sampled_magnitudes
        )
        loss_lpips = self.lpips_loss(origs, edits).mean()

        print(
            f"logits.requires_grad: {logits.requires_grad}, logits.grad_fn: {logits.grad_fn}"
        )
        print(
            f"shift_pred.requires_grad: {shift_pred.requires_grad}, shift_pred.grad_fn: {shift_pred.grad_fn}"
        )
        print(
            f"edits.requires_grad: {edits.requires_grad}, edits.grad_fn: {edits.grad_fn}"
        )
        div_loss = 0.0
        B = batch_size
        C, H, W = edits.shape[1:]
        edits_per_sample = edits.view(B, M, C, H, W)
        count = 0
        for i in range(batch_size):
            batch_edits = edits_per_sample[i]
            for a in range(M):
                for b in range(a+1, M):
                    diff = batch_edits[a] - batch_edits[b]
                    div_loss += torch.mean(diff**2)
                    count += 1
        div_loss = - div_loss / count
        div_weight = 0.5
        print(f"div_loss: {div_loss}")
        # If edits.grad_fn is None, then aux_net is not in the graph for edits.
        loss = loss_cls + loss_shift + self.lpips_weight*loss_lpips + div_weight* div_loss

        # 6) Backprop & step
        self.opt_aux.zero_grad()
        self.opt_reg.zero_grad()
        loss.backward()
        self.opt_aux.step()
        self.opt_reg.step()

        return edits, origs, flat_dirs, loss.item(), loss_cls.item(), loss_shift.item(), loss_lpips.item()

    def visualize_edirections(self, edits, origs, dirs, save_dir=None):
        """Plot one before/after pair for each direction in [0..max(dirs)]."""
        num_dirs = int(dirs.max().item()) + 1
        fig, axes = plt.subplots(num_dirs, 2, figsize=(6, 3 * num_dirs))
        for d in range(num_dirs):
            idx = (dirs == d).nonzero(as_tuple=True)[0].item()
            o = origs[idx]
            e = edits[idx]
            # rescale from [-1,1] to [0,1]
            o = ((o / 2 + 0.5).clamp(0, 1)).permute(1, 2, 0).cpu().detach().numpy()
            e = ((e / 2 + 0.5).clamp(0, 1)).permute(1, 2, 0).cpu().detach().numpy()

            axes[d, 0].imshow(o)
            axes[d, 0].set_title(f"Orig (dir={d})")
            axes[d, 0].axis("off")

            axes[d, 1].imshow(e)
            axes[d, 1].set_title(f"Edit (dir={d})")
            axes[d, 1].axis("off")

        plt.tight_layout()
        if save_dir:
            fig.savefig(save_dir, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def visualize_edirections(self, edits, origs, dirs, save_dir=None):
        """
        Plot one before/after pair for each direction in [0..max(dirs)].
        edits/origs: Tensor[(B*M), C, H, W]
        dirs:       LongTensor[(B*M,)]
        """
        num_dirs = int(dirs.max().item()) + 1
        fig, axes = plt.subplots(num_dirs, 2, figsize=(6, 3 * num_dirs))
        # ensure axes is always 2D
        if num_dirs == 1:
            axes = axes.reshape(1, 2)

        for d in range(num_dirs):
            # 1) find all positions where dirs == d
            idxs = (dirs == d).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                # no sample for this direction: skip
                continue
            idx = idxs[0].item()  # first match

            # grab original & edited
            o = origs[idx]
            e = edits[idx]

            # rescale from [-1,1] to [0,1] and move to HWC
            o = ((o / 2 + 0.5).clamp(0, 1)).permute(1, 2, 0).cpu().detach().numpy()
            e = ((e / 2 + 0.5).clamp(0, 1)).permute(1, 2, 0).cpu().detach().numpy()

            ax_orig, ax_edit = axes[d]
            ax_orig.imshow(o)
            ax_orig.set_title(f"Orig (dir={d})")
            ax_orig.axis("off")

            ax_edit.imshow(e)
            ax_edit.set_title(f"Edit (dir={d})")
            ax_edit.axis("off")

        plt.tight_layout()
        if save_dir:
            fig.savefig(save_dir, dpi=150)
            plt.close(fig)
        else:
            plt.show()