"""
resnet50_hook_mapping.py — Phase 1: UNet-ResNet50 semantic hook mapping.

Defines the exact PyTorch forward-hook registration points for extracting
feature maps from a segmentation_models_pytorch UNet with a ResNet50 encoder
(smp.Unet(encoder_name='resnet50', ...)).

Semantic boundaries
-------------------
* The ResNet50 backbone downsamples in 4 stages (strides 2, 4, 8, 16).
  The semantic "end of encoder stage k" is the LAST Bottleneck of that stage
  (encoder.layer{k}[-1]). A forward hook registered there fires AFTER the
  stage has fully finished and BEFORE the next stage's stride-2 convolution,
  giving the maximum-resolution feature map for that stage.

* The decoder in smp.Unet is a list of DecoderBlocks:
    decoder.blocks[0] -> center/bottleneck bridge (stride 16)
    decoder.blocks[1] -> stride  8 (256 -> 128 features)
    decoder.blocks[2] -> stride  4 (128 -> 64)
    decoder.blocks[3] -> stride  2 (64  -> 32)
    decoder.blocks[4] -> stride  1 (32  -> full-res)
  Registering on decoder.blocks[i] captures the output of that decoder
  block immediately after it finishes processing and before the next
  upsampling + skip-fusion step downstream.

Encoder -> feature map extraction points (before each downsampling):
    Encoder Block k  ->  encoder.layer{k}[-1]

Decoder -> feature map extraction points (after each upsampling block):
    Decoder Block k  ->  decoder.blocks[k]   (k = 1..4)
    Bridge (optional)-> decoder.blocks[0]    (bottleneck, stride 16)

Usage
-----
    from resnet50_hook_mapping import UNET_RESNET50_HOOKS, register_unet_resnet50_hooks

    model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                     in_channels=3, classes=1)

    # 1) semantic -> module-path string mapping (dict)
    print(UNET_RESNET50_HOOKS)

    # 2) register hooks; `feature_bank` receives {semantic_stage: tensor}
    feature_bank = {}
    handles = register_unet_resnet50_hooks(
        model, feature_bank,
        encoder_levels=(1, 2, 3, 4),
        decoder_levels=(1, 2, 3, 4),
        include_bridge=False,
        modality=None,            # optional label for hooks
    )
"""

from __future__ import annotations

__all__ = [
    "ENCODER_HOOK_MAP",
    "DECODER_HOOK_MAP",
    "UNET_RESNET50_HOOKS",
    "register_unet_resnet50_hooks",
    "SEMANTIC_TABLE",
]

# ---------------------------------------------------------------------------
# Semantic stage -> PyTorch module path
# ---------------------------------------------------------------------------

# Encoder: end of each ResNet50 stage (last Bottleneck), BEFORE the next
# stride-2 downsampling convolution.
ENCODER_HOOK_MAP = {
    "encoder_block_1": "encoder.layer1[-1]",   # stride 2,  256x256 -> 128x128 (for 512 input)
    "encoder_block_2": "encoder.layer2[-1]",   # stride 4
    "encoder_block_3": "encoder.layer3[-1]",   # stride 8
    "encoder_block_4": "encoder.layer4[-1]",   # stride 16
}

# Decoder: end of each DecoderBlock (after its conv/attention processing),
# BEFORE the next upsampling + skip-fusion step. blocks[0] is the stride-16
# bottleneck bridge and can optionally be treated as a separate "bridge" stage.
DECODER_HOOK_MAP = {
    "decoder_block_1": "decoder.blocks[1]",    # stride 8  (upsampling 16x8)
    "decoder_block_2": "decoder.blocks[2]",    # stride 4
    "decoder_block_3": "decoder.blocks[3]",    # stride 2
    "decoder_block_4": "decoder.blocks[4]",    # stride 1  (full resolution)
}

# Optional bridge / bottleneck (stride-16 center of the U-bottleneck).
BRIDGE_HOOK_MAP = {
    "bridge": "decoder.blocks[0]",             # stride 16 (center block)
}

UNET_RESNET50_HOOKS = {
    **ENCODER_HOOK_MAP,
    **BRIDGE_HOOK_MAP,
    **DECODER_HOOK_MAP,
}

# ---------------------------------------------------------------------------
# Human-readable table for reporting / the research log
# ---------------------------------------------------------------------------

SEMANTIC_TABLE = [
    # (Semantic Stage, Module Path, Resolution Level, Notes)
    ("Encoder Block 1", "encoder.layer1[-1]", "Stride 2",  "End of ResNet stage1 (3x Bottleneck)"),
    ("Encoder Block 2", "encoder.layer2[-1]", "Stride 4",  "End of ResNet stage2 (4x Bottleneck)"),
    ("Encoder Block 3", "encoder.layer3[-1]", "Stride 8",  "End of ResNet stage3 (6x Bottleneck)"),
    ("Encoder Block 4", "encoder.layer4[-1]", "Stride 16", "End of ResNet stage4 (3x Bottleneck)"),
    ("Bridge",          "decoder.blocks[0]",  "Stride 16", "Bottleneck center block (optional)"),
    ("Decoder Block 1", "decoder.blocks[1]",  "Stride 8",  "First upsampling decoder block"),
    ("Decoder Block 2", "decoder.blocks[2]",  "Stride 4",  "Second upsampling decoder block"),
    ("Decoder Block 3", "decoder.blocks[3]",  "Stride 2",  "Third upsampling decoder block"),
    ("Decoder Block 4", "decoder.blocks[4]",  "Stride 1",  "Final decoder block (full resolution)"),
]


# ---------------------------------------------------------------------------
# Helper: resolve a module-path string like decoder.blocks[3] to a module
# ---------------------------------------------------------------------------

def _resolve_path(model, path: str):
    """Resolve 'encoder.layer1[-1]' / 'decoder.blocks[3]' to the module object."""
    if "[-1]" in path:
        base, _ = path.rsplit("[-1]", 1)
        module = model
        for part in base.split("."):
            module = getattr(module, part)
        return module[-1]
    module = model
    for chunk in path.split("."):
        if "[" in chunk:
            name, idx = chunk.split("[")
            module = getattr(module, name)[int(idx.rstrip("]"))]
        else:
            module = getattr(module, chunk)
    return module


def register_unet_resnet50_hooks(
    model,
    feature_bank: dict,
    encoder_levels=(1, 2, 3, 4),
    decoder_levels=(1, 2, 3, 4),
    include_bridge: bool = False,
    modality: str | None = None,
):
    """Register forward hooks at the UNet-ResNet50 semantic boundaries.

    Parameters
    ----------
    model : smp.Unet
        A segmentation_models_pytorch Unet with encoder_name='resnet50'.
    feature_bank : dict
        Mutable dict; each hook writes {stage_name: feature_tensor} here.
    encoder_levels : tuple[int]
        ResNet stages to hook (1..4).
    decoder_levels : tuple[int]
        Decoder blocks to hook (1..4) — excludes the stride-16 bridge.
    include_bridge : bool
        Also hook decoder.blocks[0] (bottleneck center).
    modality : str | None
        Optional prefix for stage names in the bank (e.g. 'unet').

    Returns
    -------
    list[torch.utils.hooks.RemovableHandle]
    """
    prefix = f"{modality}." if modality else ""
    handles = []

    for level in encoder_levels:
        path = f"encoder.layer{level}[-1]"
        stage_name = f"{prefix}encoder.block{level}"
        module = _resolve_path(model, path)

        def _make_hook(name):
            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                feature_bank[name] = out.detach()
            return hook

        handles.append(module.register_forward_hook(_make_hook(stage_name)))

    if include_bridge:
        stage_name = f"{prefix}bridge"
        module = _resolve_path(model, BRIDGE_HOOK_MAP["bridge"])

        def _bridge_hook(_mod, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            feature_bank[stage_name] = out.detach()

        handles.append(module.register_forward_hook(_bridge_hook))

    for level in decoder_levels:
        path = f"decoder.blocks[{level}]"
        stage_name = f"{prefix}decoder.block{level}"
        module = _resolve_path(model, path)

        def _make_hook(name):
            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                feature_bank[name] = out.detach()
            return hook

        handles.append(module.register_forward_hook(_make_hook(stage_name)))

    return handles


if __name__ == "__main__":
    # Self-check: verify every mapped path resolves on a live smp.Unet.
    import segmentation_models_pytorch as smp

    _model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                      in_channels=3, classes=1)
    print(f"Resolved {len(UNET_RESNET50_HOOKS)} hook paths OK.")
    for _stage, _path in UNET_RESNET50_HOOKS.items():
        try:
            _m = _resolve_path(_model, _path)
            print(f"  [OK] {_stage} -> {_path} -> {type(_m).__name__}")
        except (AttributeError, IndexError) as e:
            print(f"  [FAIL] {_stage} -> {_path} -> {e}")
    print("\nSemantic table:")
    for row in SEMANTIC_TABLE:
        print("  %-18s %-22s %-10s %s" % row)