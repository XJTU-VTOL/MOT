import torch

def collate_fn(*batch):
    batch_data = batch[0]
    images = []
    labels = []
    for id, b in enumerate(batch_data):
        image, label = b
        num_label = len(label)
        batch_id = torch.ones((num_label, 1)) * id
        label = torch.cat([batch_id, label], dim=1)
        labels.append(label)
        images.append(image)

    return torch.stack(images, dim=0), torch.cat(labels, dim=0)