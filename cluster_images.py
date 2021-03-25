from pathlib import Path
from datasets.coco import CocoDetection
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy
from sklearn.cluster import KMeans
import pickle

num_epochs = 100
batch_size = 128
learning_rate = 1e-5

class CustomDataset(Dataset):
    def __init__(self, ids: np.array, x: np.array):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ids_tensor = from_numpy(ids).float().to(device)
        self.x_tensor = from_numpy(x).float().to(device)

    def __getitem__(self, index):
        return self.ids_tensor[index], self.x_tensor[index]

    def __len__(self):
        return len(self.x_tensor)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(90, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True), nn.Linear(64, 90), nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_embedding():
    # Load annotations
    coco_path = '/nfs/data3/koner/data/mscoco'
    root = Path(coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    img_folder, ann_file = PATHS['train']
    dataset = CocoDetection(img_folder, ann_file, None, None)

    targets = []
    for img_id in dataset.ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        target = dataset.coco.loadAnns(ann_ids)
        tmp = [0]*90
        for t in target:
            tmp[t['category_id']-1] += 1
        targets.append(tmp)
    targets = np.array(targets)
    ids = np.array(dataset.ids)
    targets = targets / np.max(targets)

    dataset = CustomDataset(ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Train encoder to learn embedding
    smallest_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            _, data = Variable(data).cuda()
            # ===================forward=====================
            output = model(data)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # ===================log========================
        if smallest_loss > total_loss:
            torch.save(model.state_dict(), './exp/sim_autoencoder.pth')
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    model.load_state_dict(torch.load('./exp/sim_autoencoder.pth'))
    outputs = []
    all_data = []
    all_id = []
    with torch.no_grad():
        model.eval()
        for id, data in test_dataloader:
            output = model.encoder(data)
            outputs.extend(output.detach().cpu().numpy())
            all_data.extend(data.detach().cpu().numpy())
            all_id.extend(id.detach().cpu().numpy())
    np.save('./exp/outputs.npy', np.array(outputs))
    np.save('./exp/all_data.npy', np.array(all_data))
    np.save('./exp/all_id.npy', np.array(all_id))
    print('done')

NUM_CLUSTERS = 20

def cluster():
    outputs = np.load('./exp/outputs.npy', allow_pickle=True)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, algorithm='full').fit(outputs)
    np.save('./exp/clusters.npy', kmeans.predict(outputs))

def save_cluster_ids():
    clusters = np.load('./exp/clusters.npy', allow_pickle=True)
    all_id = np.load('./exp/all_id.npy', allow_pickle=True)
    cluster_dict = {}
    id_dict = {}

    for i in range(NUM_CLUSTERS):
        cluster_dict[i] = list(all_id[clusters == i].astype(int))
    with open('./exp/cluster_dict.json', 'wb') as fp:
        pickle.dump(cluster_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    for id, cluster in zip(all_id, clusters):
        id_dict[id.astype(int)] = cluster.astype(int)
    with open('./exp/id_dict.json', 'wb') as fp:
        pickle.dump(id_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    create_embedding()
    cluster()
    save_cluster_ids()



