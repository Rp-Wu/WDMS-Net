import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from WDMS_Net import WDMS_Net


class UnnormalizedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        for class_idx in range(10):
            class_dir = os.path.join(data_dir, str(class_idx))
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return image, label
        except (OSError, IOError) as e:
            print(f"Warning: Error loading image {img_path}: {e}")
            return Image.new('RGB', (299, 299), color=0), label


class NormalizedDataset(Dataset):
    def __init__(self, subset, mean, std):
        self.subset = subset
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label


class TensorDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label


def calculate_mean_std(dataset):
    tensor_dataset = TensorDataset(dataset)
    loader = DataLoader(tensor_dataset, batch_size=64, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.numpy(), std.numpy()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', writer=None,
                fold=0):
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels_list, train_preds)

        if epoch % 5 == 0:
            for params in optimizer.param_groups:
                params['lr'] *= 0.6

        model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)

        if val_acc > best_acc:
            best_acc = val_acc

        if writer:
            writer.add_scalar(f'Fold{fold}/Train_Accuracy', train_acc, epoch)
            writer.add_scalar(f'Fold{fold}/Val_Accuracy', val_acc, epoch)
            writer.add_scalar(f'Fold{fold}/Train_Loss', train_loss, epoch)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    return best_acc


def set_seed(seed):
    """设置全局随机种子，保证 DataLoader shuffle 与模型初始化均可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    data_dir = "/18949_10fold_dataset"
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    n_splits = 10

    # 多次十折交叉验证，每次更换模型初始化种子
    seed_list = [2026,520,88]

    dataset = UnnormalizedDataset(data_dir)

    if len(dataset) == 0:
        print("Error: No images found in the dataset directory.")
        return

    labels = np.array(dataset.labels)
    # 固定数据划分种子，保证不同 init 种子下数据划分一致，便于公平比较
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    all_results = {}  # init_seed -> 各折最佳验证准确率

    for seed in seed_list:
        print(f'\n########## Init Seed = {seed} ##########')
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
            print(f'\n=== Seed {seed} | Fold {fold + 1}/{n_splits} ===')

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            print('Calculating mean and std for training set...')

            mean, std = calculate_mean_std(train_subset)
            print(f'Fold {fold + 1} - Mean: {mean}, Std: {std}')

            train_normalized = NormalizedDataset(train_subset, mean, std)
            val_normalized = NormalizedDataset(val_subset, mean, std)

            set_seed(seed)

            train_loader = DataLoader(train_normalized, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_normalized, batch_size=batch_size, shuffle=False)

            model = WDMS_Net(num_classes=10, seed=seed)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-2)

            log_dir = f'./ten_fold_log/seed_{seed}/fold_{fold + 1}'
            writer = SummaryWriter(log_dir)

            best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer,
                                   fold + 1)
            fold_accuracies.append(best_acc)

            writer.close()

            print(f'Seed {seed} | Fold {fold + 1} Best Validation Accuracy: {best_acc:.4f}')

        all_results[seed] = fold_accuracies

        print(f'\n=== Seed {seed} CV Results ===')
        print(f'Fold Accuracies: {[f"{acc:.4f}" for acc in fold_accuracies]}')
        mean_acc = np.mean(fold_accuracies)
        var_acc = np.var(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        print(f'Mean: {mean_acc:.4f} | Variance: {var_acc:.6f} | Std: {std_acc:.4f}')

    print('\n========== Overall Summary ==========')
    for seed, accs in all_results.items():
        print(f'Seed {seed:5d}: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}')
    all_means = [np.mean(accs) for accs in all_results.values()]
    print(f'\nGrand Mean across seeds: {np.mean(all_means):.4f}')
    print(f'Grand Std across seeds:  {np.std(all_means):.4f}')


if __name__ == '__main__':
    main()