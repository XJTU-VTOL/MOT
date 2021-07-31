import torch

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # Bottom left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # Bottom left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # Top right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # Top right y
    return y