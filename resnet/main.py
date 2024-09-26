from pytorch_lightning.callbacks import ModelCheckpoint
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device: ", device)

# function for inference 
def upscale_image(model, lr_image):
    model.eval()
    with torch.no_grad():
        lr_image = lr_image.unsqueeze(0)  # Add batch dimension
        recon_image = model(lr_image)
        return recon_image.squeeze(0)  # Remove batch dimension


# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',   # Metric to monitor
    filename='checkpoint',  # Filename for the best checkpoint
    save_top_k=3,          # Save only the best model
    mode='min'             # Save the model with minimum validation loss
)


if __name__ == "__main__":
    import numpy as np
    from .model import LitSRResnet, SRResnetConfig
    import pytorch_lightning as L
    from dataset import get_dataloader

    config = SRResnetConfig().default()

    _type = input("train / inference, yes if training (y/N): ")
    train_dset, val_dset = get_dataloader()
    if _type == "y":
        lit_model = LitSRResnet(config)
        trainer = L.Trainer(max_epochs=2, callbacks=[checkpoint_callback])
        trainer.fit(lit_model, train_dataloaders=train_dset, val_dataloaders=val_dset)

    else:
        checkpoint_path = input("Enter the checkpoint path: ")
        config.is_training = False
        model = LitSRResnet(config).load_from_checkpoint(checkpoint_path if checkpoint_path is not None else "checkpoint.ckpt")
        model.eval()



