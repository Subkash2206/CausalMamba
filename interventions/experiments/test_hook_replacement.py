"""
Isolated PyTorch capability test: forward hook output replacement.

Verifies that a forward hook can:
    1. Replace a module's output before it reaches the next module.
    2. Propagate the modified tensor downstream.
    3. Be removed cleanly to restore original behaviour.

Network: Conv2d(3, 8, 3) → ReLU → Conv2d(8, 1, 3)

No datasets, no training, no SpectralMamba dependency.
"""

import torch
import torch.nn as nn


class TinyNet(nn.Module):
    """Minimal CNN for hook testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


def print_stats(label: str, tensor: torch.Tensor) -> None:
    """Print summary statistics for a tensor."""
    print(
        f"  {label:20s}: "
        f"shape={list(tensor.shape)}, "
        f"min={tensor.min().item():+.4f}, "
        f"max={tensor.max().item():+.4f}, "
        f"mean={tensor.mean().item():+.4f}, "
        f"norm={tensor.norm().item():.4f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # Initialise network and a fixed input.
    # ------------------------------------------------------------------
    model = TinyNet().to(device)
    model.eval()

    # Fixed input so results are reproducible across runs.
    torch.manual_seed(42)
    x = torch.randn(2, 3, 16, 16, device=device)

    # Container to capture intermediate values from the hook.
    captured = {}  # type: dict[str, torch.Tensor]
    hook_handle = None

    # ------------------------------------------------------------------
    # Phase 1 — Run without hook (baseline).
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1 — Baseline (no hook)")
    print("=" * 60)

    with torch.no_grad():
        # Manually run step-by-step to inspect every stage.
        out_conv1 = model.conv1(x)
        out_relu = model.relu(out_conv1)
        out_final = model.conv2(out_relu)

    print_stats("conv1 output", out_conv1)
    print_stats("relu output", out_relu)
    print_stats("final output", out_final)
    baseline_final = out_final.clone()
    print()

    # ------------------------------------------------------------------
    # Phase 2 — Register hook that replaces conv1 output with zeros.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 2 — Hook: replace conv1 output with zeros")
    print("=" * 60)

    def zeroing_hook(module: nn.Module, input: tuple[torch.Tensor],
                     output: torch.Tensor) -> torch.Tensor:
        """Replace the module output with zeros of the same shape."""
        # Capture the *original* output for comparison.
        captured["original_conv1"] = output.detach().clone()
        # Return the replacement.
        return torch.zeros_like(output)

    # Register the hook on the first Conv2d.
    hook_handle = model.conv1.register_forward_hook(zeroing_hook)

    with torch.no_grad():
        out_conv1_hooked = model.conv1(x)
        out_relu_hooked = model.relu(out_conv1_hooked)
        out_final_hooked = model.conv2(out_relu_hooked)

    print_stats("captured original conv1", captured["original_conv1"])
    print_stats("hooked conv1 output", out_conv1_hooked)
    print_stats("hooked relu output", out_relu_hooked)
    print_stats("hooked final output", out_final_hooked)

    # ------------------------------------------------------------------
    # Verification 1 — Does the downstream ReLU receive the modified tensor?
    # ------------------------------------------------------------------
    relu_is_zero = (out_relu_hooked.abs().max().item() < 1e-6)
    print(f"\n  [{'PASS' if relu_is_zero else 'FAIL'}] "
          f"ReLU receives zeros: {relu_is_zero}")

    # ------------------------------------------------------------------
    # Verification 2 — Does the final network output change?
    # ------------------------------------------------------------------
    final_diff = (out_final_hooked - baseline_final).abs().max().item()
    final_changed = final_diff > 1e-4
    print(f"  [{'PASS' if final_changed else 'FAIL'}] "
          f"Final output changed (max|diff| = {final_diff:.4f})")
    print()

    # ------------------------------------------------------------------
    # Phase 3 — Remove hook and verify original behaviour is restored.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 3 — Hook removed, verify restoration")
    print("=" * 60)

    assert hook_handle is not None
    hook_handle.remove()

    with torch.no_grad():
        out_conv1_restored = model.conv1(x)
        out_relu_restored = model.relu(out_conv1_restored)
        out_final_restored = model.conv2(out_relu_restored)

    print_stats("restored conv1 output", out_conv1_restored)
    print_stats("restored relu output", out_relu_restored)
    print_stats("restored final output", out_final_restored)

    # ------------------------------------------------------------------
    # Verification 3 — Original behaviour restored.
    # ------------------------------------------------------------------
    conv1_restored = (out_conv1_restored - out_conv1).abs().max().item() < 1e-6
    final_restored = (out_final_restored - baseline_final).abs().max().item() < 1e-6
    print(f"\n  [{'PASS' if conv1_restored else 'FAIL'}] "
          f"conv1 output restored: {conv1_restored}")
    print(f"  [{'PASS' if final_restored else 'FAIL'}] "
          f"Final output restored: {final_restored}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    all_pass = relu_is_zero and final_changed and conv1_restored and final_restored
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # Clean up.
    del model, x


if __name__ == "__main__":
    main()