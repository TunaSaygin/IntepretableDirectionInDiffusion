import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split      # <-- fix import
from diffusion_model_t_cond import DiffusionModel
from tqdm import tqdm
# — Hyper-parameters —
num_directions = 16
num_samples_per_direction = 800
device = "cuda"
mag = 0.5                                        # whatever magnitude you want
batch_size = 32

# — Instantiate your model & load weights —
dm = DiffusionModel(
    sample_dim=(1, 3, 256, 256),
    model_name="google/ddpm-ema-celebahq-256",
    num_directions=16,
    target_timestep=800,
    duration_of_change=100,
    num_inference_steps=100,
    device=device,
    use_resnet=True,
)
ckpt = torch.load("/home/ubuntu/unsupervised_dir/result_cirriculumn/diffusion_full_ckpt_199.pt", map_location=device)
dm.aux_net.load_state_dict(ckpt["aux_net"])
dm.regressor.load_state_dict(ckpt["regressor"])
dm.custom_diffusion.unet_edited.eval()
dm.aux_net.eval()
with torch.no_grad():
    # — Generate your (image, label) pairs —
    images, labels = [], []
    for d in tqdm(range(num_directions)):
        for _ in tqdm(range(num_samples_per_direction)):
            # sample a fresh latent z  — shape (1, 3, 256, 256)
            z = torch.randn(1, 3, 256, 256, device=device)
            with torch.no_grad():
                img = dm.custom_diffusion.generate_image(z, direction_idx=d, magnitude=torch.Tensor([mag]).to("cuda"), inject_aux=True) 
                # img: tensor in [0,1] or [-1,1], shape (1,3,256,256)
            images.append(img.squeeze(0).cpu())    # store as (3,256,256)
            labels.append(d)

# — Dataset & DataLoader —
class DirectionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels = images, labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img, lbl = self.images[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, lbl

dataset = DirectionDataset(
    images, labels,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
)
# 1) split dataset
n_total = len(dataset)
n_test  = int(0.2 * n_total)
n_train = n_total - n_test
train_ds, test_ds = random_split(dataset, [n_train, n_test])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

# (Re)initialize your reconstructor, optimizer, criterion as before:
reconstructor = torchvision.models.resnet18(pretrained=False)
reconstructor.fc = nn.Linear(reconstructor.fc.in_features, num_directions)
reconstructor = reconstructor.to(device)
opt = torch.optim.Adam(reconstructor.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
patience = 3
epochs_since_improvement = 0

for epoch in range(1, 21):  # max 20 epochs
    # — TRAINING STEP —
    reconstructor.train()
    train_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = reconstructor(imgs)
        loss = criterion(logits, lbls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # — VALIDATION STEP —
    reconstructor.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = reconstructor(imgs)
            preds  = logits.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    val_acc = correct / total

    print(
        f"Epoch {epoch:2d}  "
        f"train_loss={train_loss:.4f}  "
        f"val_acc={val_acc*100:.2f}%  "
        f"(best {best_val_acc*100:.2f}%, patience {epochs_since_improvement}/{patience})"
    )

    # — EARLY STOPPING LOGIC —
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_since_improvement = 0
        torch.save(reconstructor.state_dict(), "best_rca.pth")
    else:
        epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(f"No improvement in {patience} epochs—stopping early.")
            break
