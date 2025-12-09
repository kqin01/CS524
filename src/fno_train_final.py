import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import torch.fft as fft



########################################
# USER CONFIG
########################################
DATA_ROOT   = "/home/kqin/OpenFOAM/kqin-7/quickFNO5000"
IMS_DIR     = "/home/kqin/OpenFOAM/kqin-7/ims"
U_DIR       = "/home/kqin/OpenFOAM/kqin-7/run/data2000Da1/U"

NUM_IMAGES  = 5000
PE_LIST     = [0, 10]  # Pe choices
U_SCALE     = 10

EPOCHS      = 300
BATCH       = 64
LR          = 1e-4



########################################
# DATASET
########################################
class PorousVelocityDataset(Dataset):
    def __init__(self, ims_dir, u_dir, split, pe_list, u_scale, num_images):

        self.ims_dir = ims_dir
        self.u_dir   = u_dir
        self.pe_list = pe_list
        self.u_scale = u_scale

        all_ids = np.arange(num_images)

        n_train = int(num_images * 0.8)
        n_dev   = int(num_images * 0.1)

        if split == "train":
            self.img_ids = all_ids[:n_train]
        elif split == "dev":
            self.img_ids = all_ids[n_train:n_train+n_dev]
        elif split == "test":
            self.img_ids = all_ids[n_train+n_dev:]
        else:
            raise ValueError("split must be train/dev/test")

        # Filter samples that have U files
        self.samples = []
        for img_id in self.img_ids:
            for pe in self.pe_list:
                u_file = os.path.join(self.u_dir, f"{img_id}_U_Pe{pe}.txt")
                if os.path.exists(u_file) and os.path.getsize(u_file) > 5:
                    self.samples.append((img_id, pe))

        print(f"[{split}] Loaded {len(self.samples)} samples (Pe={self.pe_list})")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_id, pe = self.samples[idx]

        # load batched .pt file (100 images per file)
        batch_start = (img_id // 100) * 100
        batch_file  = os.path.join(self.ims_dir, f"{batch_start}-{batch_start+99}.pt")
        img_batch = torch.load(batch_file)

        image = img_batch[img_id % 100].float()
        if image.ndim == 2:
            image = image.unsqueeze(0)

        # normalize the binary image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # load Ux, Uy
        u_file = os.path.join(self.u_dir, f"{img_id}_U_Pe{pe}.txt")
        u_full = np.loadtxt(u_file)
        u_vec = torch.tensor(u_full[:2], dtype=torch.float32) / self.u_scale

        # normalized Pe
        # ----- General Pe normalization -----
        den = max(self.pe_list) - min(self.pe_list)
        if den < 1e-8:
            # Only one Pe or all equal -> constant 0
            pe_norm = torch.tensor([0.0], dtype=torch.float32)
        else:
            pe_norm = torch.tensor([(pe - min(self.pe_list)) / den], dtype=torch.float32)

        return image, pe_norm, u_vec



########################################
# FNO LAYER
########################################
class FourierLayer2D(nn.Module):
    """
    Basic low-frequency spectral convolution (FNO)
    """
    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w

        # trainable complex weights
        self.weight_real = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_h, modes_w) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_h, modes_w) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = fft.rfft2(x)  # (B, C, H, W//2+1)

        out_ft = torch.zeros(
            B, self.out_channels, H, W//2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W//2 + 1)

        out_ft[:, :, :mh, :mw] = torch.einsum(
            "bchw,cihw->bihw",
            x_ft[:, :, :mh, :mw],
            (self.weight_real + 1j * self.weight_imag)
        )

        return fft.irfft2(out_ft, s=(H, W))



########################################
# FNO MODEL
########################################
class PorousFNO_U(nn.Module):
    """
    Light FNO for image -> scalar regression (Ux, Uy)
    """
    def __init__(self, modes_h=16, modes_w=16, width=32):
        super().__init__()

        # project input to width channels
        self.fc0 = nn.Conv2d(1, width, 1)

        self.fno1 = FourierLayer2D(width, width, modes_h, modes_w)
        self.fno2 = FourierLayer2D(width, width, modes_h, modes_w)
        self.fno3 = FourierLayer2D(width, width, modes_h, modes_w)

        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.act = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(width + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x, pe):
        x = self.fc0(x)

        x1 = self.fno1(x) + self.w1(x)
        x1 = self.act(x1)

        x2 = self.fno2(x1) + self.w2(x1)
        x2 = self.act(x2)

        x3 = self.fno3(x2) + self.w3(x2)
        x3 = self.act(x3)

        g = self.pool(x3).view(x3.size(0), -1)
        g = torch.cat([g, pe], dim=1)
        return self.fc(g)



########################################
# COLOR MAP
########################################
def get_pe_colors(pe_values):
    pe_values = sorted(list(set(pe_values)))
    base = {0: "#80e5e5", 10: "#7f7f7f"}
    cmap = plt.get_cmap("tab20")
    colors, j = {}, 0
    for pe in pe_values:
        colors[pe] = base.get(pe, cmap(j % cmap.N))
        j += 1
    return colors



########################################
# PLOT: pred vs true
########################################
def plot_pred_vs_true_split(model, loaders, names, device, u_scale, save_prefix):

    model.eval()

    for loader, name in zip(loaders, names):

        preds, trues, pes_true = [], [], []

        with torch.no_grad():
            for image, pe_norm, target in loader:
                pred = model(image.to(device), pe_norm.to(device))

                preds.append(pred.cpu().numpy() * u_scale)
                trues.append(target.numpy() * u_scale)
                pes_true.append(pe_norm.numpy().reshape(-1) * max(PE_LIST))

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        pes_true = np.concatenate(pes_true)

        unique_pes = sorted(np.unique(pes_true))
        pe_colors = get_pe_colors(unique_pes)

        # Ux
        plt.figure(figsize=(6, 6))
        for pe in unique_pes:
            m = (pes_true == pe)
            plt.scatter(trues[m,0], preds[m,0], s=14, alpha=0.6,
                        color=pe_colors[pe], label=f"Pe={int(pe)}")

        mn = min(trues[:,0].min(), preds[:,0].min())
        mx = max(trues[:,0].max(), preds[:,0].max())
        plt.plot([mn, mx], [mn, mx], "k--")
        plt.xlabel("True Ux")
        plt.ylabel("Pred Ux")
        plt.grid(); plt.legend(); plt.tight_layout()
        plt.savefig(f"{save_prefix}_{name}_Ux.png", dpi=150)
        plt.close()

        # Uy
        plt.figure(figsize=(6, 6))
        for pe in unique_pes:
            m = (pes_true == pe)
            plt.scatter(trues[m,1], preds[m,1], s=14, alpha=0.6,
                        color=pe_colors[pe], label=f"Pe={int(pe)}")

        mn = min(trues[:,1].min(), preds[:,1].min())
        mx = max(trues[:,1].max(), preds[:,1].max())
        plt.plot([mn, mx], [mn, mx], "k--")
        plt.xlabel("True Uy")
        plt.ylabel("Pred Uy")
        plt.grid(); plt.legend(); plt.tight_layout()
        plt.savefig(f"{save_prefix}_{name}_Uy.png", dpi=150)
        plt.close()



########################################
# TRAINING
########################################
def train_full():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    train_set = PorousVelocityDataset(IMS_DIR, U_DIR, "train", PE_LIST, U_SCALE, NUM_IMAGES)
    dev_set   = PorousVelocityDataset(IMS_DIR, U_DIR, "dev",   PE_LIST, U_SCALE, NUM_IMAGES)
    test_set  = PorousVelocityDataset(IMS_DIR, U_DIR, "test",  PE_LIST, U_SCALE, NUM_IMAGES)

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=8, pin_memory=True)
    dev_loader   = DataLoader(dev_set,   batch_size=BATCH)
    test_loader  = DataLoader(test_set,  batch_size=BATCH)


    model = PorousFNO_U().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_losses, dev_losses = [], []

    print("---- Training ----")
    for ep in range(EPOCHS):

        model.train()
        total, cnt = 0, 0

        for img, pe, u in train_loader:
            img, pe, u = img.to(device), pe.to(device), u.to(device)

            pred = model(img, pe)
            loss = loss_fn(pred, u)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * len(img)
            cnt   += len(img)

        train_losses.append(total/cnt)

        # -------- DEV ----------
        model.eval()
        total, cnt = 0, 0
        with torch.no_grad():
            for img, pe, u in dev_loader:
                pred = model(img.to(device), pe.to(device))
                loss = loss_fn(pred, u.to(device))
                total += loss.item() * len(img)
                cnt   += len(img)

        dev_losses.append(total/cnt)

        print(f"Epoch {ep+1}/{EPOCHS} | Train={train_losses[-1]:.5f} | Dev={dev_losses[-1]:.5f}")


    # -------- TEST PHASE ----------
    model.eval()
    total, cnt = 0, 0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for img, pe, u in test_loader:
            pred = model(img.to(device), pe.to(device))

            all_preds.append(pred.cpu().numpy() * U_SCALE)
            all_trues.append(u.numpy() * U_SCALE)

            loss = loss_fn(pred, u.to(device))
            total += loss.item() * len(img)
            cnt   += len(img)

    test_loss = total/cnt
    print("Final Test Loss =", test_loss)




    # ---- Save trained model ----
    model_path = os.path.join(DATA_ROOT, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")





    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    np.savetxt(os.path.join(DATA_ROOT, "predictions.txt"), all_preds)
    np.savetxt(os.path.join(DATA_ROOT, "truth.txt"), all_trues)

    rms = np.sqrt(np.mean((all_preds - all_trues)**2, axis=0))
    with open(os.path.join(DATA_ROOT, "metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"RMS Ux: {rms[0]}\n")
        f.write(f"RMS Uy: {rms[1]}\n")

    # Loss curve
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(dev_losses, label="Dev")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_ROOT, "loss_curve.png"), dpi=150)
    plt.close()

    # Scatter plots
    plot_pred_vs_true_split(
        model,
        [train_loader, dev_loader, test_loader],
        ["train", "dev", "test"],
        device,
        U_SCALE,
        os.path.join(DATA_ROOT, "UxUy")
    )



###############################################################
# MAIN
###############################################################
if __name__ == "__main__":
    train_full()