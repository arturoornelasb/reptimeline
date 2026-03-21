"""
Binary Autoencoder for MNIST — reptimeline second backend demo.

Architecture: 784 -> 256 -> 128 -> 32 sigmoid+STE -> 128 -> 256 -> 784
Trains in ~2 minutes, saves checkpoints every 2 epochs.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryAE(nn.Module):
    """Autoencoder with a binary bottleneck using Straight-Through Estimator."""

    def __init__(self, input_dim=784, hidden=256, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid(),
        )

    def binarize(self, z):
        """Straight-Through Estimator: hard threshold forward, sigmoid gradient backward."""
        z_sig = torch.sigmoid(z)
        z_hard = (z_sig > 0.5).float()
        return z_hard - z_sig.detach() + z_sig  # STE trick

    def encode_binary(self, x):
        """Encode to binary code (for inference)."""
        z = self.encoder(x)
        return (torch.sigmoid(z) > 0.5).float()

    def forward(self, x):
        z = self.encoder(x)
        z_bin = self.binarize(z)
        recon = self.decoder(z_bin)
        return recon, z_bin


def train_binary_ae(
    output_dir="results/mnist_bae/checkpoints",
    epochs=10,
    save_every=2,
    batch_size=256,
    lr=1e-3,
    bottleneck=32,
    diversity_weight=0.01,
    device="cuda",
):
    """Train Binary AE on MNIST and save checkpoints.

    Args:
        output_dir: Where to save model checkpoints.
        epochs: Total training epochs.
        save_every: Save checkpoint every N epochs.
        bottleneck: Number of binary bits.
        diversity_weight: Weight for diversity loss (prevents dead bits).
        device: 'cuda' or 'cpu'.
    """
    from torchvision import datasets, transforms

    os.makedirs(output_dir, exist_ok=True)

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_data = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Model
    model = BinaryAE(input_dim=784, hidden=256, bottleneck=bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Save initial (random) checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, "model_step0.pt"))
    print(f"Saved checkpoint: epoch 0 (random init)")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_div = 0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device)
            recon, z_bin = model(x)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")

            # Diversity loss: push mean activation toward 0.5
            div_loss = (z_bin.mean(dim=0) - 0.5).pow(2).mean()

            loss = recon_loss + diversity_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_div += div_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_div = total_div / n_batches

        # Bit utilization
        model.eval()
        with torch.no_grad():
            sample = next(iter(train_loader))[0][:500].to(device)
            codes = model.encode_binary(sample)
            active_bits = (codes.mean(dim=0) > 0.02).sum().item()

        print(f"  Epoch {epoch:>2d}/{epochs}  loss={avg_loss:.4f}  "
              f"recon={avg_recon:.4f}  div={avg_div:.4f}  "
              f"active_bits={active_bits}/{bottleneck}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            path = os.path.join(output_dir, f"model_step{epoch}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved checkpoint: epoch {epoch}")

    print(f"\nTraining complete. Checkpoints in {output_dir}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Binary AE on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bottleneck", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/mnist_bae/checkpoints")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    train_binary_ae(
        output_dir=args.output, epochs=args.epochs,
        bottleneck=args.bottleneck, device=args.device,
    )
