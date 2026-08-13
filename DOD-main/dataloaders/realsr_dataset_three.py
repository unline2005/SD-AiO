import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import random

class PairedSROnlineTxtDataset(Dataset):
    def __init__(self, split='train', args=None, tasks=None,
                 batch_task_ratios=None, placeholder_token="*",
                 template="a photo of a {}"):
        super().__init__()
        self.args = args
        self.split = split
        self.tasks = tasks or {
            'dehazing': 'OTS',
            'deraining': 'rain100L',
            'denoising': ['BSDWED15', 'BSDWED25', 'BSDWED50']
        }
        self.batch_task_ratios = batch_task_ratios or {'dehazing': 3, 'deraining': 1, 'denoising': 2}
        self.task_data = {}
        self.task_iterators = {}  # 用于遍历任务数据的迭代器
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.placeholder_token = placeholder_token
        self.template = template

        base_dir = '/tn/xmu/data/'
        self.data_dir = os.path.join(base_dir)

        # 读取任务数据
        for task_name, task_dirs in self.tasks.items():
            if isinstance(task_dirs, list):
                gt_files, lq_files = [], []
                for task_dir in task_dirs:
                    task_path = os.path.join(self.data_dir, task_dir)
                    gt_dir = os.path.join(task_path, 'GT')
                    lq_dir = os.path.join(task_path, 'LQ')
                    gt_files.extend(sorted([f for f in glob.glob(os.path.join(gt_dir, '*')) if self._is_image(f)]))
                    lq_files.extend(sorted([f for f in glob.glob(os.path.join(lq_dir, '*')) if self._is_image(f)]))
            else:
                task_path = os.path.join(self.data_dir, task_dirs)
                gt_dir = os.path.join(task_path, 'GT')
                lq_dir = os.path.join(task_path, 'LQ')
                gt_files = sorted([f for f in glob.glob(os.path.join(gt_dir, '*')) if self._is_image(f)])
                lq_files = sorted([f for f in glob.glob(os.path.join(lq_dir, '*')) if self._is_image(f)])

            assert len(gt_files) == len(lq_files), f"Task {task_name} has mismatched GT and LQ counts"
            self.task_data[task_name] = {
                'gt': gt_files,
                'lq': lq_files,
                'indices': list(range(len(gt_files)))  # 维护索引
            }
            self._reset_iterator(task_name)  # 初始化任务迭代器

    def _reset_iterator(self, task_name):
        """ 重新初始化任务数据迭代器（用于保证所有数据遍历） """
        indices = self.task_data[task_name]['indices'].copy()
        np.random.shuffle(indices)  # 洗牌数据
        self.task_iterators[task_name] = iter(indices)

    def __len__(self):
        """ 数据集的总长度 = 最长任务的数据量 """
        return max([len(d['indices']) for d in self.task_data.values()])

    def _get_task_sample(self, task_name):
        """ 取出任务中的下一个样本 """
        try:
            task_idx = next(self.task_iterators[task_name])
        except StopIteration:  
            self._reset_iterator(task_name)  # 重新洗牌，确保所有数据被遍历
            task_idx = next(self.task_iterators[task_name])

        gt_path = self.task_data[task_name]['gt'][task_idx]
        lq_path = self.task_data[task_name]['lq'][task_idx]
        gt_img = Image.open(gt_path).convert('RGB')
        lq_img = Image.open(lq_path).convert('RGB')

        if self.split == 'train':
            # 检查尺寸是否小于256×256，若是则先resize
            width, height = gt_img.size
            if height < 256 or width < 256:
                gt_img = F.resize(gt_img, size=[256, 256], interpolation=transforms.InterpolationMode.BICUBIC)
                lq_img = F.resize(lq_img, size=[256, 256], interpolation=transforms.InterpolationMode.BICUBIC)

            # 随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(gt_img, (256, 256))
            gt_img = F.crop(gt_img, i, j, h, w)
            lq_img = F.crop(lq_img, i, j, h, w)

            # 水平翻转
            if random.random() < 0.5:
                gt_img = F.hflip(gt_img)
                lq_img = F.hflip(lq_img)

            # 垂直翻转
            if random.random() < 0.5:
                gt_img = F.vflip(gt_img)
                lq_img = F.vflip(lq_img)


        else:
            gt_img = F.center_crop(gt_img, (256, 256))
            lq_img = F.center_crop(lq_img, (256, 256))

        return {
            "output_pixel_values": self._preprocess(gt_img),
            "conditioning_pixel_values": self._preprocess(lq_img),
            "task_name": task_name,
            "neg_prompt": getattr(self.args, 'neg_prompt', '')
        }

    def __getitem__(self, idx):
        """ 每个 batch 需要按比例从不同任务取样 """
        samples = []
        for task_name, num_samples in self.batch_task_ratios.items():
            for _ in range(num_samples):
                samples.append(self._get_task_sample(task_name))
        return samples  # 返回 batch_size=6 份样本

    def _preprocess(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return F.normalize(img, self.mean, self.std)

    def _is_image(self, filename):
        ext = os.path.splitext(filename)[-1].lower()
        return ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']


def custom_collate_fn(batch):
    """ 合并 batch，确保数据结构正确 """
    all_samples = [sample for sublist in batch for sample in sublist]
    batch_dict = {key: torch.stack([s[key] for s in all_samples]) for key in all_samples[0] if isinstance(all_samples[0][key], torch.Tensor)}
    batch_dict["task_name"] = [s["task_name"] for s in all_samples]
    batch_dict["neg_prompt"] = [s["neg_prompt"] for s in all_samples]
    return batch_dict


# **测试代码**
if __name__ == '__main__':
    dataset = PairedSROnlineTxtDataset(
        split='train',
        batch_task_ratios={'dehazing': 3, 'deraining': 1, 'denoising': 2}
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    for epoch in range(2):  # 模拟2个epoch
        print(f"Epoch {epoch + 1} Start")
        for batch in dataloader:
            print(f"Batch Task Distribution: {batch['task_name']}")
        print(f"Epoch {epoch + 1} End\n")
