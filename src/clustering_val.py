import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np


def extract_dense_features_global(model, dataloader, device, sample_fraction=1.0):
    """
    Извлекает dense признаки для всего датасета: из последней карты признаков, преобразованной в форму [B*H*W, C].
    """
    model.eval()
    all_features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feat_maps = model.backbone(images)[-1]  # [B, C, H, W]
            feat_maps = F.normalize(feat_maps, p=2, dim=1)
            B, C, H, W = feat_maps.shape
            features = feat_maps.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
            if sample_fraction < 1.0:
                num_samples = int(features.size(0) * sample_fraction)
                idx = torch.randperm(features.size(0))[:num_samples]
                features = features[idx]
            all_features.append(features.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)
    return all_features

def global_clustering_dbscan(model, dataloader, device, eps=0.5, min_samples=5, sample_fraction=1.0):
    features = extract_dense_features_global(model, dataloader, device, sample_fraction)
    print(f"Total extracted features: {features.shape[0]}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features)
    
    unique_clusters = set(cluster_labels)
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    
    if len(unique_clusters) < 2:
        print("Not enough clusters (>=2 required) in global clustering.")
        return None, None, None, cluster_labels
    
    sil = silhouette_score(features, cluster_labels)
    db_index = davies_bouldin_score(features, cluster_labels)
    ch_score = calinski_harabasz_score(features, cluster_labels)
    return sil, db_index, ch_score, cluster_labels

def extract_dense_features_per_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
        feat_map = model.backbone(image)[-1]     # [1, C, H, W]
        feat_map = F.normalize(feat_map, p=2, dim=1)
        _, C, H, W = feat_map.shape
        features = feat_map.view(1, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
    return features.cpu().numpy()

def extract_dense_features_per_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
        feat_map = model.backbone(image)[-1]     # [1, C, H, W]
        feat_map = F.normalize(feat_map, p=2, dim=1)
        _, C, H, W = feat_map.shape
        features = feat_map.view(1, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
    return features.cpu().numpy()

def per_image_clustering_dbscan(model, dataloader, device, eps=0.5, min_samples=5):
    model.eval()
    sil_scores, db_scores, ch_scores = [], [], []
    image_count = 0
    for images, _ in dataloader:
        for i in range(images.size(0)):
            features = extract_dense_features_per_image(model, images[i], device)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)
            unique_clusters = set(labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)
            if len(unique_clusters) < 2:
                continue
            try:
                sil = silhouette_score(features, labels)
                db = davies_bouldin_score(features, labels)
                ch = calinski_harabasz_score(features, labels)
                sil_scores.append(sil)
                db_scores.append(db)
                ch_scores.append(ch)
                image_count += 1
            except Exception as e:
                print(f"Error on image {image_count}: {e}")
                continue
    if len(sil_scores) == 0:
        print("No image produced enough clusters for per-image evaluation.")
        return None, None, None
    avg_sil = np.mean(sil_scores)
    avg_db = np.mean(db_scores)
    avg_ch = np.mean(ch_scores)
    # print(f"Processed {image_count} images. Avg Silhouette: {avg_sil:.4f}, Avg Davies-Bouldin: {avg_db:.4f}, Avg Calinski-Harabasz: {avg_ch:.4f}")
    return avg_sil, avg_db, avg_ch