import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def read_csv(csv_file):
    """ 读取 CSV 文件，返回 epochs、loss、class_error、AP 指标 """
    if not os.path.exists(csv_file):
        return [], [], [], [[] for _ in range(6)]

    epochs, losses, class_errors, coco_ap_metrics = [], [], [], [[] for _ in range(6)]
    
    with open(csv_file, mode='r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if row is not []:
                epochs.append(int(row[0]))
                losses.append(float(row[1]))
                class_errors.append(float(row[2]))
                for i in range(6):
                    coco_ap_metrics[i].append(float(row[3 + i]))

    return np.array(epochs), np.array(losses), np.array(class_errors), np.array(coco_ap_metrics)

def plot_results(csv_file,output_dir,existing_epochs,losses, class_errors, aps,suffix="base"):
    """ 绘制损失、误差率和 COCO 评估曲线 """

    fig, ax1 = plt.subplots()

    # 绘制损失曲线
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(existing_epochs, losses, label="Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # 绘制误差率曲线
    ax2 = ax1.twinx()
    ax2.set_ylabel("Class Error", color="tab:red")
    ax2.plot(existing_epochs, class_errors, label="Class Error", color="tab:red", linestyle="dashed")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title(f"Loss & Class Error over Epochs {suffix}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_error_curve_{suffix}.png"))
    plt.close()

    # 绘制 COCO AP 指标
    plt.figure()
    plt.plot(existing_epochs, aps[0], label="AP", color="tab:green")
    plt.plot(existing_epochs, aps[1], label="AP50", color="tab:blue")
    plt.plot(existing_epochs, aps[2], label="AP75", color="tab:purple")
    plt.plot(existing_epochs, aps[3], label="APS (Small)", color="tab:orange")
    plt.plot(existing_epochs, aps[4], label="APM (Medium)", color="tab:brown")
    plt.plot(existing_epochs, aps[5], label="APL (Large)", color="tab:pink")

    plt.xlabel("Epoch")
    plt.ylabel("COCO AP")
    plt.title(f"COCO AP Metrics over Epochs {suffix}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"coco_ap_metrics_curve_{suffix}.png"))
    plt.close()

if __name__ == '__main__':
    suffix_list = ['base','old','new','all']
    for suffix in suffix_list:
        csv_file = os.path.join('outputs', f"evaluation_results_{suffix}.csv")
        epochs, losses, class_errors, aps = read_csv(csv_file)
        plot_results(csv_file,'outputs',epochs,losses, class_errors, aps,suffix)

