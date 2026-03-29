from matplotlib import pyplot as plt
import torch
import config
from refinement import refine_output

def plot_array(imgs, title_list=None):
    n_imgs = len(imgs)
    fig, axs = plt.subplots(1, n_imgs, figsize=(6 * n_imgs, 6))
    for idx in range(len(imgs)):
        axs[idx].imshow(imgs[idx], cmap='gray' if imgs[idx].ndim == 2 else None)
        title = title_list[idx] if title_list else f"IMAGE {idx}"
        axs[idx].set_title(title)
        axs[idx].axis("off")
    plt.tight_layout()
    plt.show()

def debug_single_chip_visualization(model, dataset, prototypes, idx=82):
    data, label, _ = dataset[idx]
    
    # Forward a single chip through the model
    coarse_logits, feat, _, _ = model(data.unsqueeze(0).cuda())
    data = data.unsqueeze(0).cuda()

    _refined_dict = refine_output(
        coarse_logits=coarse_logits,
        prototypes=prototypes,
        dino_feats=feat,
        image_rgb=data / 255.0
    )

    plt.imshow(torch.argmax(_refined_dict["logits_s1"][0], dim=0).cpu().numpy(), cmap='Accent', vmin=0, vmax=6)
    plt.title("S1 Prediction (Prototype Similarity)")
    plt.show()

    if not config.only_s1:
        plt.imshow(torch.argmax(_refined_dict["logits_s2"][0], dim=0).cpu().numpy(), cmap='Accent', vmin=0, vmax=6)
        plt.title("S2 Prediction (LP Refinement)")
        plt.show()

    plt.imshow(label, cmap='Accent', vmin=0, vmax=6)
    plt.title("Ground Truth")
    plt.show()
