from my_dataset import get_dataset_and_loader_ASV2021_LA, get_dataset_and_loader_ASV2021_inner
from torch.utils.data import ConcatDataset, DataLoader

def get_true(tag):
    datasets = []
    dataloaders = []
    for ds_name in tag:
        if ds_name=="ASV2021_LA":
            ds,dl = get_dataset_and_loader_ASV2021_LA("real")
        if ds_name=="ASV2021_inner":
            ds,dl = get_dataset_and_loader_ASV2021_inner("real")
        datasets.append(ds)
        dataloaders.append(dl)
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=16, shuffle=True, num_workers=4)
    print(f"总数据量: {len(combined_dataset)}")
    return combined_dataset,loader


