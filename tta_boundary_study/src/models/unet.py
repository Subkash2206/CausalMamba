import segmentation_models_pytorch as smp

def get_unet(encoder='resnet50', pretrained=True):
    encoder_weights = 'imagenet' if pretrained else None
    
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None
    )
    
    return model
