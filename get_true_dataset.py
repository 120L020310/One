from my_dataset import get_dataset_and_loader_ASV2021_LA, get_dataset_and_loader_ASV2021_inner, get_dataset_and_loader_ASV5
from torch.utils.data import ConcatDataset, DataLoader

def get_true(tag,stage,bs):
    datasets = []
    dataloaders = []
    for ds_name in tag:
        if ds_name=="ASV2021_LA":
            ds,dl = get_dataset_and_loader_ASV2021_LA("real",stage)
        if ds_name=="ASV2021_inner":
            ds,dl = get_dataset_and_loader_ASV2021_inner("real",stage)
        if ds_name=="ASV5":
            ds,dl = get_dataset_and_loader_ASV5("real",stage)
        datasets.append(ds)
        dataloaders.append(dl)
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=bs, shuffle=True, num_workers=4)
    print(f"总数据量: {len(combined_dataset)}")
    return combined_dataset,loader

# ds,dl = get_true(["ASV2021_LA","ASV2021_inner","ASV5"])
# print("ibb")
