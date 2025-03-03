import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

def visualize_clustering_on_image(model, image, eps, min_samples):

    image_batch = image.to('cuda').unsqueeze(0)  # [1, C, H, W]
    
    model.eval()
    with torch.no_grad():
        # [1, C_feat, H_feat, W_feat]
        feat_map = model.backbone(image_batch)[-1]
    
    _, C_feat, H_feat, W_feat = feat_map.shape
    feat_map_norm = F.normalize(feat_map, p=2, dim=1)
    
    #[C_feat, H_feat*W_feat] -> [H_feat*W_feat, C_feat]
    features = feat_map_norm.squeeze(0).view(C_feat, -1).permute(1, 0).cpu().numpy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features)
    # [H_feat, W_feat]
    cluster_map = cluster_labels.reshape(H_feat, W_feat)
    
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    orig_img = image.cpu().permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    im = axes[1].imshow(cluster_map, cmap='jet')
    axes[1].set_title(f"DBSCAN Clusters (n = {n_clusters})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1])
    
    return fig