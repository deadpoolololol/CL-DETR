"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from util import box_ops
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

def train_one_epoch_incremental(model: torch.nn.Module, old_model: torch.nn.Module, ref_loss_overall_coef,
                    criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    old_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        ref_outputs = old_model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0) # 原输出尺寸
        
        # 根据旧模型的输出来生成伪标签
        ref_results = postprocessors['bbox'](ref_outputs, orig_target_sizes, 10, distillation=True) # 取前K=10个
        
        # 计算旧模型的损失
        ref_loss_dict = criterion(outputs, ref_results, enable_aux=False)
        ref_weight_dict = criterion.ref_weight_dict
        ref_losses = sum(ref_loss_dict[k] * ref_weight_dict[k] for k in ref_loss_dict.keys() if k in ref_weight_dict)

        # 使用 IoU（交并比）计算来决定哪些伪标签可以被加入到训练数据
        for img_idx in range(len(targets)):
            include_list = []
            for ref_box_idx in range(len(ref_results[img_idx]['boxes'])):
                this_ref_box = ref_results[img_idx]['boxes'][ref_box_idx]
                this_ref_box = torch.reshape(this_ref_box, (1, -1))
                include_this_pseudo_label = True
                for target_box_idx in range(len(targets[img_idx]['boxes'])):
                    this_target_box = targets[img_idx]['boxes'][target_box_idx]
                    this_target_box = torch.reshape(this_target_box, (1, -1))
                    iou, union = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(this_ref_box), box_ops.box_cxcywh_to_xyxy(this_target_box))
                    if iou >= 0.7: # λ = 0.7
                        include_this_pseudo_label = False
                include_list.append(include_this_pseudo_label)

            # 伪标签会被添加到 targets 中  
            targets[img_idx]['boxes'] = torch.cat((targets[img_idx]['boxes'], ref_results[img_idx]['boxes'][include_list]), 0)
            targets[img_idx]['labels'] = torch.cat((targets[img_idx]['labels'], ref_results[img_idx]['labels'][include_list]), 0)          

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        #losses += ref_loss_overall_coef*ref_losses

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_base(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    print_freq = 50
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator



@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,epoch_num,suffix="base"):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    loss_list, class_error_list = [], []
    coco_ap_metrics = []  # 用于存储 COCO 评估指标
    print_freq = 50
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        
        total_loss = sum(loss_dict_reduced_scaled.values()).item()
        class_error = loss_dict_reduced['class_error'].item()
        
        metric_logger.update(loss=total_loss, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=class_error)

        # 记录损失和误差
        loss_list.append(total_loss)
        class_error_list.append(class_error)

        # 计算 COCO 评估结果
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, orig_target_sizes, orig_target_sizes)
            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 记录 COCO 评估所有 AP 指标
    if coco_evaluator is not None and 'bbox' in postprocessors.keys():
        coco_ap_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
        stats.update({
            "AP": coco_ap_stats[0],
            "AP50": coco_ap_stats[1],
            "AP75": coco_ap_stats[2],
            "APS": coco_ap_stats[3],
            "APM": coco_ap_stats[4],
            "APL": coco_ap_stats[5]
        })
        coco_ap_metrics.append(coco_ap_stats)  # 存储所有 AP 指标

    # 保存 CSV
    csv_file = os.path.join(output_dir, f"evaluation_results_{suffix}.csv")
    epochs, losses, class_errors, aps = read_csv(csv_file)
    save_results_to_csv(csv_file, loss_list, class_error_list, coco_ap_metrics,epochs,epoch_num)

    # 绘制结果
    plot_results(csv_file,output_dir, loss_list, class_error_list, coco_ap_metrics,epochs,losses, class_errors, aps,epoch_num,suffix)

    return stats, coco_evaluator


def get_epoch_num(csv_file):
    existing_epochs = set()
    # 如果文件存在，先读取已有的 epoch 记录
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头
            for row in reader:
                if row and row[0].isdigit():  # 确保是数字
                    existing_epochs.add(int(row[0]))

    return existing_epochs

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

def save_results_to_csv(csv_file, loss_list, class_error_list, coco_ap_metrics,existing_epochs,epoch_num):
    """ 读取已保存数据，避免重复写入 """
    loss_avg = sum(loss_list) / len(loss_list)
    class_error_avg = sum(class_error_list) / len(class_error_list)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if len(existing_epochs) == 0 :
            writer.writerow(["Epoch", "Loss", "Class Error", "AP", "AP50", "AP75", "APS", "APM", "APL"])  # 表头

        if epoch_num  not in existing_epochs:
            writer.writerow([
                epoch_num, loss_avg, class_error_avg,
                coco_ap_metrics[0][0], coco_ap_metrics[0][1], coco_ap_metrics[0][2], 
                coco_ap_metrics[0][3], coco_ap_metrics[0][4], coco_ap_metrics[0][5]
            ])



def plot_results(csv_file,output_dir, loss_list, class_error_list, coco_ap_metrics,existing_epochs,losses, class_errors, aps,epoch_num,suffix="base"):
    """ 绘制损失、误差率和 COCO 评估曲线 """

    loss_avg = sum(loss_list) / len(loss_list)
    class_error_avg = sum(class_error_list) / len(class_error_list)


    np.append(existing_epochs,epoch_num)
    np.append(losses,loss_avg)
    np.append(class_errors,class_error_avg)

    for i in range(6):
        np.append(aps[i],coco_ap_metrics[0][i])

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
    plt.title("Loss & Class Error over Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_error_curve_{suffix}.png"))
    plt.close()

    # 绘制 COCO AP 指标
    # coco_ap_metrics = np.array(coco_ap_metrics)
    plt.figure()
    plt.plot(existing_epochs, aps[0], label="AP", color="tab:green")
    plt.plot(existing_epochs, aps[1], label="AP50", color="tab:blue")
    plt.plot(existing_epochs, aps[2], label="AP75", color="tab:purple")
    plt.plot(existing_epochs, aps[3], label="APS (Small)", color="tab:orange")
    plt.plot(existing_epochs, aps[4], label="APM (Medium)", color="tab:brown")
    plt.plot(existing_epochs, aps[5], label="APL (Large)", color="tab:pink")

    plt.xlabel("Epoch")
    plt.ylabel("COCO AP")
    plt.title("COCO AP Metrics over Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"coco_ap_metrics_curve_{suffix}.png"))
    plt.close()


