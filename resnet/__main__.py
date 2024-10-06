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
    from model import LitSRResnet, SRResnetConfig
    import pytorch_lightning as L
    import os
    import matplotlib.pyplot as plt
    from dataset import get_dataloader

    # config = SRResnetConfig().default()
    config = SRResnetConfig(
        hidden_channel=3,
        lr=1e-3,
        is_training=True
    )

    _type = input("train / inference, yes if training (y/N): ")
    train_dset,val_dset =get_dataloader(batch_size=1, num_workers=8, input_px=128, output_px=262)
    if _type == "y":
        lit_model = LitSRResnet()
        trainer = L.Trainer(max_epochs=2, callbacks=[checkpoint_callback])
        trainer.fit(lit_model, train_dataloaders=train_dset, val_dataloaders=val_dset)

    else:
        checkpoint_path = input("Enter the checkpoint path: ")
        config.is_training = False
        if checkpoint_path == "":
            print("setting default checkpoint path - checkpoint.ckpt")
            checkpoint_path = "model.ckpt"
        #checkpoint = torch.load(checkpoint_path, weights_only=True)
        # check if path is valid
        if not os.path.exists(checkpoint_path):
            print("Invalid path")
            exit(1)
        litmodel = LitSRResnet.load_from_checkpoint(checkpoint_path)
        litmodel.eval()
        for param in litmodel.parameters():
            param.requires_grad = False

        show_image = lambda x: plt.imshow(x[0].permute(1, 2, 0).cpu().numpy())
        for x, target in val_dset:
            x = x.to(config.device)
            show_image(x)
            plt.savefig("input.png")
            show_image(target)
            plt.savefig("target.png")
            print(f"{x.shape=}, {target.shape=}")
            y = litmodel.model(x)
            print(y.shape)
            show_image(y) # show the image
            plt.imshow(x[0].permute(1, 2, 0).cpu().numpy())
            plt.savefig("generated.png")
            break

        print("Model loaded successfully")



