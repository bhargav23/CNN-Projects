from torchvision import transforms
class Transforms:
  def train_transforms(self,transforms_list=[]):
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)

  def test_transforms(self,transforms_list=[]):
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)