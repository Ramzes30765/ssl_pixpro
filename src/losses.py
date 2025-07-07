import torch.nn.functional as F


def pixpro_loss(p1, p2, y1, y2, tau=0.2, eps=1e-8):
    p1_flat = F.normalize(p1.flatten(2), dim=1, eps=eps)
    p2_flat = F.normalize(p2.flatten(2), dim=1, eps=eps)
    y1_flat = F.normalize(y1.flatten(2), dim=1, eps=eps)
    y2_flat = F.normalize(y2.flatten(2), dim=1, eps=eps)
    loss1 = -F.cosine_similarity(p1_flat, y2_flat.detach(), dim=1).mean() / tau 
    loss2 = -F.cosine_similarity(p2_flat, y1_flat.detach(), dim=1).mean() / tau 
    return 0.5 * (loss1 + loss2)