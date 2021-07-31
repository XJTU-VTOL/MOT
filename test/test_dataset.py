from dataset import ToyDataset
import unittest
from torch.utils.data import DataLoader
from dataset.util import collate_fn

class TestDataset(unittest.TestCase):
    def test_toy_dataset(self):
        dataset_config = dict(
            image_size = (608, 1088)
        )
        dataset = ToyDataset(dataset_config)
        loader = DataLoader(dataset, batch_size=8, num_workers=4, collate_fn=collate_fn)
        for batch in loader:
            image, target = batch
            print(image.shape)
            print(target.shape)
            break

if __name__ == '__main__':
    # verbosity=*：默认是1；设为0，则不输出每一个用例的执行结果；2-输出详细的执行结果
    unittest.main(verbosity=2)