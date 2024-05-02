import torch.nn.functional as F
import faiss
import numpy as np
from ttts.utils.infer_utils import load_model
from ttts.vqvae.dataset import VQGANDataset, VQVAECollater, BucketSampler
from ttts.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
import json
from torch.utils.data import DataLoader
import torch
import faiss
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

n_cpu = 40
def train_index(features):
    npys = []
    big_npy = features.detach().cpu().numpy()
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 1000:
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=4096,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            # yield "\n".join(infos)
    exp_dir = 'ttts/vqvae'
    # np.save(f"{exp_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(192, "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    # faiss.write_index(
    #     index,
    #     "%s/trained_IVF%s_Flat_nprobe_%s.index"
    #     % (exp_dir, n_ivf, index_ivf.nprobe),
    # )
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe),
    )

cfg_path='ttts/vqvae/config_v3.json'
cfg = json.load(open(cfg_path))
hps = HParams(**cfg)
device='cuda'
vqvae_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v3/2024-04-29-03-34-14/model-118.pt'
model = load_model('vqvae', vqvae_path, 'ttts/vqvae/config_v3.json', device)
dataset = VQGANDataset(hps)
train_sampler = BucketSampler(
            dataset, hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000,
                1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,],
            shuffle=True,)
collate_fn=VQVAECollater()
dataloader = DataLoader(
    dataset,
    num_workers=8,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
    persistent_workers=True,
    batch_sampler=train_sampler,
    prefetch_factor=16,)
features =[]
c=0
for data in tqdm(dataloader):
    wav = data['wav']
    wav_lengths = data['wav_lengths']
    spec = spectrogram_torch(wav, 1024, 320, 1024, center=False).squeeze(0).to(device)
    spec_length = torch.LongTensor([x//320 for x in wav_lengths]).to(device)
    with torch.no_grad():
        outs = model.extract_feature(spec,spec_length)
    for i, feature in enumerate(outs):
        features.append(feature[:,:spec_length[i]//4])

features = torch.cat(features,dim=1).transpose(0,1)
train_index(features)

# index = faiss.read_index(file_index)
# big_npy = index.reconstruct_n(0, index.ntotal)


# npy = feat[0].cpu().numpy()
# _, I = index.search(npy, 1)
# npy = big_npy[I.squeeze()]

# feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device))