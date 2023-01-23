from dataset.dataset import *
from config.config import *
from torch.utils.data import DataLoader

def get_dataloader(attributes):
    attr = AttributesDataset(attributes)
    train_dataset = FashionDataset('./train.csv', attr, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZES, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = FashionDataset('./val.csv', attr, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZES, shuffle=False, num_workers=NUM_WORKERS)
    
    test_dataset = FashionDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_dataloader, val_dataloader, test_dataloader
    