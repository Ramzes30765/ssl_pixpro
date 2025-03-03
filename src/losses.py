import torch.nn.functional as F


def pixpro_loss(p1, p2, y1, y2):
    # Flatten по пространственным измерениям: [B, proj_dim, H, W] -> [B, proj_dim, H*W]
    p1_flat = p1.flatten(2)
    p2_flat = p2.flatten(2)
    y1_flat = y1.flatten(2)
    y2_flat = y2.flatten(2)
    # Вычисляем негативное косинусное сходство
    loss1 = -F.cosine_similarity(p1_flat, y2_flat.detach(), dim=1).mean()
    loss2 = -F.cosine_similarity(p2_flat, y1_flat.detach(), dim=1).mean()
    return 0.5 * (loss1 + loss2)