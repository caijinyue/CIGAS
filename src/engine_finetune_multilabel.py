import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Iterable, Optional
from timm.data import Mixup
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, confusion_matrix
import util.misc as misc
import util.lr_sched as lr_sched

def configure_tent_model(model):
    """Configuration for TENT: enable grad for Norm layers, disable for others."""
    model.eval()
    params = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            for param_name, param in module.named_parameters():
                param.requires_grad = True
                params.append(param)
                names.append(f'{name}.{param_name}')
        else:
            pass
    for name, param in model.named_parameters():
        if name not in names:
            param.requires_grad = False
    return params

def check_requires_grad(model):
    print('Params with requires_grad=True:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'  {name}')

def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def _should_generate_heatmap(args, mode, num_class):
    return getattr(args, 'heatmap', False) and mode == 'test' and (not args.multi_label) and (num_class > 1)

def _heatmap_run_tag(args):
    return str(args.tta).upper() if getattr(args, 'tta', None) else 'noTENT'

def _get_vit_patch_grid(backbone):
    if not hasattr(backbone, 'patch_embed') or not hasattr(backbone.patch_embed, 'num_patches'):
        raise ValueError('Feature-map export requires a ViT backbone with patch embeddings.')
    num_patches = backbone.patch_embed.num_patches
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f'Unsupported non-square patch layout for feature-map export: {num_patches} patches.')
    return grid_size

def _tokens_to_feature_map(tensor, height, width):
    if tensor.ndim != 3:
        raise ValueError(f'Expected ViT features with shape [B, N, C], got {tuple(tensor.shape)}')
    num_tokens = tensor.shape[1]
    expected_patch_tokens = height * width
    if num_tokens == expected_patch_tokens + 1:
        tensor = tensor[:, 1:, :]
    elif num_tokens != expected_patch_tokens:
        raise ValueError(f'Unexpected token count for feature-map reshape: got {num_tokens}, expected {expected_patch_tokens} or {expected_patch_tokens + 1}')
    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)

def _get_last_block_feature_context(model):
    backbone = _unwrap_model(model)
    if not hasattr(backbone, 'blocks') or len(backbone.blocks) == 0:
        raise ValueError('Feature-map export requires a ViT-like backbone with transformer blocks.')
    if not hasattr(backbone, 'head') or not hasattr(backbone.head, 'weight'):
        raise ValueError('Feature-map export requires a linear classification head with weights.')
    return (backbone, backbone.blocks[-1], _get_vit_patch_grid(backbone))

def _denormalize_image(sample):
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=sample.device, dtype=sample.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=sample.device, dtype=sample.dtype).view(3, 1, 1)
    return torch.clamp(sample * std + mean, 0.0, 1.0)

def _normalize_map_for_overlay(value_map):
    abs_max = np.percentile(np.abs(value_map), 99)
    if abs_max <= 1e-08:
        abs_max = 1e-08
    return (np.clip(value_map / abs_max, -1.0, 1.0), abs_max)

def _resize_signed_map(value_map, target_hw):
    target_h, target_w = target_hw
    image = Image.fromarray(((value_map + 1.0) * 127.5).astype(np.uint8))
    image = image.resize((target_w, target_h), resample=Image.BICUBIC)
    return np.asarray(image).astype(np.float32) / 127.5 - 1.0

def _save_feature_maps(feature_maps, classifier_weight, samples, filenames, heatmap_dir, targets=None, probs=None, preds=None):
    raw_dir = os.path.join(heatmap_dir, 'raw_npz')
    os.makedirs(raw_dir, exist_ok=True)
    weight = classifier_weight.detach().float().cpu()
    class1_weight = weight[1]
    margin_weight = weight[1] - weight[0]
    for idx, (feature_map, sample, filename) in enumerate(zip(feature_maps, samples, filenames)):
        if targets is not None and preds is not None:
            gt_label = int(targets[idx].detach().cpu().item())
            pred_label = int(preds[idx].detach().cpu().item())
            if pred_label != gt_label:
                continue
        stem = os.path.splitext(os.path.basename(str(filename)))[0]
        raw_path = os.path.join(raw_dir, f'{stem}_feature_map_raw.npz')
        try:
            feature_map = feature_map.detach().float().cpu()
            class1_map = torch.einsum('c,chw->hw', class1_weight, feature_map).numpy()
            margin_map = torch.einsum('c,chw->hw', margin_weight, feature_map).numpy()
            normalized_margin_map, margin_abs_max = _normalize_map_for_overlay(margin_map)
            image_np = _denormalize_image(sample.detach()).permute(1, 2, 0).cpu().numpy()
            normalized_margin_map_resized = _resize_signed_map(normalized_margin_map, image_np.shape[:2])
            raw_payload = {'feature_map_chw': feature_map.numpy().astype(np.float32), 'class1_weighted_map': class1_map.astype(np.float32), 'class_margin_map': margin_map.astype(np.float32), 'class_margin_map_normalized': normalized_margin_map.astype(np.float32), 'class_margin_map_normalized_resized': normalized_margin_map_resized.astype(np.float32), 'class_margin_abs_max': np.float32(margin_abs_max), 'image_rgb': image_np.astype(np.float32), 'input_tensor_chw': sample.detach().cpu().numpy().astype(np.float32), 'target_class': np.int64(1)}
            if targets is not None:
                raw_payload['gt_label'] = np.int64(gt_label)
            if preds is not None:
                raw_payload['pred_label'] = np.int64(pred_label)
            if probs is not None:
                raw_payload['class1_score'] = np.float32(probs[idx][1].detach().cpu().item())
            np.savez_compressed(raw_path, **raw_payload)
        except Exception as exc:
            print(f'[heatmap] Failed to save feature map for {filename}: {exc}')

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float=0, mixup_fn: Optional[Mixup]=None, log_writer=None, args=None):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = (20, args.accum_iter)
    optimizer.zero_grad()
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    for data_iter_step, (samples, targets, filenames) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples, targets = (samples.to(device, non_blocking=True), targets.to(device, non_blocking=True))
        if args.multi_label:
            targets = targets.float()
            if mixup_fn is not None:
                print('Warning: Mixup disabled in multi-label mode.')
                mixup_fn = None
        else:
            targets = targets.long()
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group['lr'])
            max_lr = max(max_lr, group['lr'])
        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            ' We use epoch_1000x as the x-axis in tensorboard.\n            This calibrates different curves when batch size changes.\n            '
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer, suffix=None):
    """Evaluate the model."""
    suffix = epoch if suffix is None else suffix
    metric_logger = misc.MetricLogger(delimiter='  ')
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    generate_heatmap = _should_generate_heatmap(args, mode, num_class)
    heatmap_dir = None
    feature_context = None
    feature_hook_handle = None
    captured_features = []
    if generate_heatmap:
        run_tag = _heatmap_run_tag(args)
        heatmap_dir = os.path.join(args.output_dir, args.task, f'feature_maps_internal_test_{run_tag}_epoch{suffix:03}')
        os.makedirs(heatmap_dir, exist_ok=True)
        feature_context = _get_last_block_feature_context(model)
        backbone, target_layer, grid_size = feature_context

        def _capture_last_block_features(module, inputs, output):
            captured_features.clear()
            captured_features.append(_tokens_to_feature_map(output.detach(), grid_size, grid_size).cpu())
        feature_hook_handle = target_layer.register_forward_hook(_capture_last_block_features)
    true_labels, pred_labels, pred_probs = ([], [], [])
    image_names = []
    is_tent = args.tta == 'tent'
    optimizer_tent = None
    if is_tent:
        print(f'[{mode}] Configuring TENT: Enabling gradients for Norm layers...')
        tent_params = configure_tent_model(model)
        if args.tent_episodic:
            print(f'[{mode}] TENT Mode: Episodic (Resets every batch)')
        else:
            print(f'[{mode}] TENT Mode: Online (Continuous update)')
            optimizer_tent = torch.optim.Adam(tent_params, lr=args.tent_lr, betas=(0.9, 0.999))
        print(f'[{mode}] TENT Optimizer initialized. Trainable params: {len(tent_params)}')
    else:
        model.eval()
    if args.multi_label:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    class_names = ['enlarged vertical cup-to-disc ratio', 'glaucomatous disc cupping', 'laminar-dot sign within the cup', 'ISNT rule violation', 'neuro-retinal rim thinning/notching', 'beta-zone peripapillary atrophy', 'optic disc pallor relative to surrounding retina', 'nasalization of central retinal vessels', 'glaucomatous disc haemorrhages', 'vessel bayonetting at disc margin', 'wedge-shaped RNFL defect', 'macula & background retinal lesion']
    if len(class_names) != num_class:
        print(f'Note: Using default class names (expected {len(class_names)} classes, got {num_class})')
        class_names = [f'class_{i}' for i in range(num_class)]
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target, filenames = (batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2])
        cached_state_dict = None
        context = torch.enable_grad() if is_tent else torch.no_grad()
        with context:
            with torch.cuda.amp.autocast():
                try:
                    if args.tta == 'tent':
                        import copy
                        cached_state_dict = None
                        with torch.cuda.amp.autocast(enabled=False):
                            images = images.float()
                            if args.tent_episodic:
                                optimizer_tent = torch.optim.Adam(tent_params, lr=args.tent_lr, betas=(0.9, 0.999))
                                cached_state_dict = copy.deepcopy(model.state_dict())
                            tent_steps = args.tent_steps
                            for _ in range(tent_steps):
                                outputs = model(images)
                                if args.multi_label:
                                    p = torch.sigmoid(outputs)
                                    p = torch.clamp(p, min=1e-07, max=1 - 1e-07)
                                    entropy = -(p * p.log() + (1 - p) * (1 - p).log())
                                    loss_tent = entropy.mean()
                                else:
                                    p = F.softmax(outputs, dim=1)
                                    p = torch.clamp(p, min=1e-07, max=1 - 1e-07)
                                    entropy = -(p * p.log()).sum(dim=1)
                                    loss_tent = entropy.mean()
                                optimizer_tent.zero_grad()
                                loss_tent.backward()
                                optimizer_tent.step()
                        output = model(images)
                    else:
                        output = model(images)
                except Exception as e:
                    print(f'Error during forward: {e}')
                    print(filenames)
                    raise
            loss = criterion(output, target)
        metric_logger.update(loss=loss.item())
        if args.multi_label:
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(preds.detach().cpu().numpy())
            pred_probs.extend(probs.detach().cpu().numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(preds.detach().cpu().numpy())
            pred_probs.extend(probs.detach().cpu().numpy())
        image_names.extend(filenames)
        if feature_context is not None and captured_features:
            backbone, _, _ = feature_context
            _save_feature_maps(captured_features[0], backbone.head.weight, images, filenames, heatmap_dir, targets=target, probs=probs, preds=preds)
        if is_tent and args.tent_episodic and (cached_state_dict is not None):
            model.load_state_dict(cached_state_dict)
    predictions_path = os.path.join(args.output_dir, args.task, f'image_predictions_{mode}_epoch{suffix:03}.csv')
    with open(predictions_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        if args.multi_label:
            header = ['image_path'] + [f'{name}_prob' for name in class_names]
        else:
            header = ['image_path'] + [f'{name}_prob' for name in class_names] + ['predicted_class']
        writer.writerow(header)
        for name, prob in zip(image_names, pred_probs):
            if args.multi_label:
                row = [name] + [f'{p:.8f}' for p in prob]
            else:
                pred_class_idx = int(np.argmax(prob))
                pred_class_name = class_names[pred_class_idx]
                row = [name] + [f'{p:.8f}' for p in prob] + [pred_class_name]
            writer.writerow(row)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)
    if args.multi_label:
        print(f'[{mode}] Total samples in evaluation: {len(true_labels)}')
        pos_per_class = true_labels.sum(axis=0)
        print(f'[{mode}] Positive samples per class: {pos_per_class}')
        print(f'[{mode}] Total positive labels: {pos_per_class.sum()}')
        hamming = hamming_loss(true_labels, pred_labels)
        accuracy = accuracy_score(true_labels.flatten(), pred_labels.flatten())
        jaccard = jaccard_score(true_labels, pred_labels, average='macro', zero_division=0)
        average_precision = average_precision_score(true_labels, pred_probs, average='macro')
        f1 = f1_score(true_labels, pred_labels, zero_division=0, average='macro')
        precision = precision_score(true_labels, pred_labels, zero_division=0, average='macro')
        recall = recall_score(true_labels, pred_labels, zero_division=0, average='macro')
        valid_classes = pos_per_class > 0
        if valid_classes.sum() == 0:
            roc_auc = 0.0
        elif valid_classes.sum() == 1:
            valid_idx = np.where(valid_classes)[0][0]
            roc_auc = roc_auc_score(true_labels[:, valid_idx], pred_probs[:, valid_idx])
        else:
            roc_auc = roc_auc_score(true_labels[:, valid_classes], pred_probs[:, valid_classes], average='macro')
        score = (f1 + roc_auc) / 2
        print(f"val loss: {metric_logger.meters['loss'].global_avg}")
        print(f'F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n Average Precision: {average_precision:.4f}, Score: {score:.4f}')
    else:
        print(f'[{mode}] Total samples in evaluation: {len(true_labels)}')
        accuracy = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels)
        if num_class == 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
            specificity = tn / (tn + fp) if tn + fp > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = sensitivity
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            try:
                roc_auc = roc_auc_score(true_labels, pred_probs[:, 1])
            except ValueError:
                roc_auc = 0.0
            print(f'[{mode}] Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')
            hamming = hamming_loss(true_labels, pred_labels)
            jaccard = jaccard_score(true_labels, pred_labels)
            average_precision = average_precision_score(true_labels, pred_probs[:, 1])
            kappa = cohen_kappa_score(true_labels, pred_labels)
        else:
            f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
            precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
            try:
                roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro')
            except ValueError:
                roc_auc = 0.0
        score = (f1 + roc_auc) / 2
        print(f"val loss: {metric_logger.meters['loss'].global_avg}")
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Score: {score:.4f}')
    if log_writer:
        for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'score'], [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
    metric_logger.synchronize_between_processes()
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'score'])
        wf.writerow([metric_logger.meters['loss'].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, score])
    if feature_hook_handle is not None:
        feature_hook_handle.remove()
    return ({k: meter.global_avg for k, meter in metric_logger.meters.items()}, score)
