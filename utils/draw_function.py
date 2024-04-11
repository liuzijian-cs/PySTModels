import matplotlib.pyplot as plt
import os
from utils.base_function import Color, print_log


def draw_record(save_path, epochs, record):
    plt.figure(figsize=(16, 12))
    epoch = range(epochs)
    # 设置标题和标签的字体大小
    title_fontsize = 24
    label_fontsize = 22
    ticks_fontsize = 20

    # Train & Valid Loss:
    plt.plot(epoch, record["train"]["loss"], label='Train Loss', color='blue', linewidth=5, linestyle='-')
    plt.plot(epoch, record["valid"]["loss"], label='Validation Loss', color='red', linewidth=5, linestyle='-')
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.title('Training and Validation Loss', fontsize=title_fontsize)
    plt.xticks(ticks=range(0, epochs, 5), fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=label_fontsize, frameon=False)
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()
    # Train Loss:
    plt.figure(figsize=(16, 12))
    plt.plot(epoch, record["train"]["loss"], label='Train Loss', color='blue', linewidth=5, linestyle='-')
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.title('Training Loss', fontsize=title_fontsize)
    plt.xticks(ticks=range(0, epochs, 5), fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=label_fontsize, frameon=False)
    plt.savefig(os.path.join(save_path, 'loss_train.png'))
    plt.close()
    # Valid Loss:
    plt.figure(figsize=(16, 12))
    plt.plot(epoch, record["valid"]["loss"], label='Valid Loss', color='blue', linewidth=5, linestyle='-')
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.title('Validation Loss', fontsize=title_fontsize)
    plt.xticks(ticks=range(0, epochs, 5), fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=label_fontsize, frameon=False)
    plt.savefig(os.path.join(save_path, 'loss_valid.png'))
    plt.close()

# def draw_predictions(args, model_path, data_type, ):
#     assert data_type in ['train', 'valid', 'test', 'all']
#     Task = task_dict[args.task].Task(args, model_dict, data_dict)
#     Task.load_model(model_path)
#     Task.draw_predictions(data_type)






# if __name__ == '__main__':
#     args = args_parser()
#     model_path = "./model_save/epoch-040.pth"
#     draw_predictions(model_path, 'train')
