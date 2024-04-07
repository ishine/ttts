import argparse
import functools
import os
import torch
import torchaudio.functional as F
import torchaudio
import json
from torch.multiprocessing import Process, set_start_method
from tqdm import tqdm
from ttts.utils.data_utils import spectrogram_torch,HParams
from torch.multiprocessing import Pool

from ttts.classifier.infer import read_jsonl
from ttts.utils.utils import find_audio_files, get_paths_with_cache
from concurrent.futures import ProcessPoolExecutor
from ttts.utils.infer_utils import load_model


def extract_vq(file_paths, max_workers, device_id):
    # 将文件路径分成 max_workers 份
    chunk_size = len(file_paths) // max_workers
    file_chunks = [file_paths[i * chunk_size: (i + 1) * chunk_size] for i in range(max_workers)]
    processes = []
    # 创建一个进程池
    for i in range(max_workers):
        process = Process(target=process_vq, args=(file_chunks[i], device_id))
        processes.append(process)
        process.start()
    # 等待所有进程完成
    for process in processes:
        process.join()
    # 过滤掉返回None的结果
    # results = [result for result in results if result is not None]

def process_vq(paths, device_id):
    hps = HParams(**json.load(open('ttts/vqvae/config.json')))
    model_path = '/home/hyc/tortoise_plus_zh/ttts/vqvae/logs/v2/G_128163.pth'
    vqvae = load_model('vqvae', model_path, 'ttts/vqvae/config.json', f'cuda:{device_id}')
    torch.cuda.set_device(device_id)
    for path in tqdm(paths):
        wav_path = path
        try:
            wav,sr = torchaudio.load(wav_path)
            if wav.shape[0] > 1:
                wav = wav[0].unsqueeze(0)
            wav = wav.cuda()
            wav32k = F.resample(wav, sr, 32000)
            wav32k = wav32k[:,:int(hps.data.hop_length * 2 * (wav32k.shape[-1]//hps.data.hop_length//2))]
            wav = torch.clamp(wav32k, min=-1.0, max=1.0)
        except Exception as e:
            print(path)
            print(e)
            return
        try:
            with torch.no_grad():
                spec = spectrogram_torch(wav, hps.data.filter_length,
                        hps.data.hop_length, hps.data.win_length, center=False)
                code = vqvae.extract_code(wav, spec).squeeze(0).squeeze(0)
        except Exception as e:
            print(path)
            print(e)
            return
        outp = path+'.vq.pth'
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        torch.save(code.tolist(), outp)
    return

if __name__ == '__main__':
    # 设置启动方法，根据你的系统环境选择'spawn'或'forkserver'
    set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='ttts/datasets/data_temp.jsonl')
    args = parser.parse_args()
    paths = read_jsonl(args.json_path)
    paths = [os.path.join('/home/hyc/tortoise_plus_zh', path['path']) for path in paths]
    
    # 假设有4张GPU卡可用
    num_gpus = 4
    num_threads = 12  # 每个GPU卡上的线程数
    device_ids = [4,5,6,7]  # GPU设备ID列表

    # 分配每个进程使用的GPU设备
    per_gpu_file_paths = [paths[i::num_gpus] for i in range(num_gpus)]
    processes = []

    # 创建并启动进程
    for i, device_id in enumerate(device_ids):
        process = Process(target=extract_vq, args=(per_gpu_file_paths[i], num_threads, device_id))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    # # 收集结果
    # results = []
    # for i in range(num_gpus):
    #     results.extend(processes[i].results)  # 假设每个进程都有一个results属性来存储结果