import torch


def save_checkpoint(epoch, model, optimizer,best_accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy':best_accuracy
    }, path)


# 加载模型和优化器状态
def load_checkpoint(model, optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    print(f"Checkpoint loaded, starting at epoch {start_epoch}.")
    return start_epoch,best_accuracy