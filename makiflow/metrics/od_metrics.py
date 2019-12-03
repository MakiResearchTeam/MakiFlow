from __future__ import absolute_import
import numpy as np
from tqdm import tqdm
from makiflow.metrics.od_utils import compute_tps, parse_dicts, nms, clear_filtered_preds, merge_nms_result


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def mAP(pred_boxes, pred_cs, pred_ps, true_boxes, true_cs, iou_th=0.5):
    """
    Parameters
    ----------
    pred_boxes : list
        List of ndarrays of shape [total_preds, 4]. Contains predicted bboxes for one image.
    pred_cs : list
        List of ndarrays of shape [total_preds]. Contains classes for `pred_boxes`.
    pred_ps : list
        List of ndarrays of shape [total_preds]. Contains confidences for `pred_cs`.
    true_boxes : list
        List of ndarrays of shape [total_preds, 4]. Contains true bboxes for one image.
    true_cs : list
        List of ndarrays of shape [total_preds]. Contains classes for `true_boxes`.

    Returns
    -------
    precision : ndarray
        Precision for each class
    recall : ndarray
        Recall for each class.
    average precision : ndarray
        Average precision for each class.
    f1-score : ndarray
        F1-score for each class.
    unique_classes : ndarray
        Array of unique classes found during mAP calculation.
    """
    num_images = len(pred_boxes)
    all_tps = []
    for i in range(num_images):
        all_tps += [compute_tps(
            pred_boxes=pred_boxes[i],
            pred_classes=pred_cs[i],
            true_boxes=true_boxes[i],
            true_classes=true_cs[i],
            iou_th=iou_th
        )]

    all_tps = np.concatenate(all_tps, axis=0)
    pred_cs = np.concatenate(pred_cs, axis=0)
    pred_ps = np.concatenate(pred_ps, axis=0)
    true_cs = np.concatenate(true_cs, axis=0)

    p, r, ap, f1, unique_classes = ap_per_class(
        tp=all_tps,
        conf=pred_ps,
        pred_cls=pred_cs,
        target_cls=true_cs
    )
    return p, r, ap, f1, unique_classes


def mAP_maki_supported(sdd_preds, iou_threshold, conf_threshold, test_dict, name2class):
    """
    A shortcut for calculation mAP.
    Parameters
    ----------
    sdd_preds : list
        List of lists [confidences, locs]. In other words, list of the ssdmodel predictions.

    iou_threshold
    conf_threshold
    test_dict
    name2class

    Returns
    -------

    """
    # PROCESS THE SSD PREDICTIONS
    filtered_preds = []
    for confidences, localisations in sdd_preds:
        for image_confs, image_locs in zip(confidences, localisations):
            # bboxes, classes, confidences
            # `bboxes` is a list of ndarrays
            # `classes` is a list of ints
            # `confidences` is a list of floats
            filtered_preds += [nms(image_confs, image_locs, conf_threshold=conf_threshold, iou_threshold=iou_threshold)]
    # Clear the NMS results from empty predictions
    filtered_preds = clear_filtered_preds(filtered_preds)
    # Convert NMS results to separate lists of numpy arrays
    pred_boxes = []
    pred_cs = []
    pred_ps = []
    for filtered_pred in filtered_preds:
        boxes, classes, confs = merge_nms_result(filtered_pred)
        pred_boxes += [boxes]
        pred_cs += [classes]
        pred_ps += [confs]

    # PROCESS THE GROUND TRUE LABELS
    true_boxes, true_classes = parse_dicts(test_dict, name2class=name2class)

    # COMPUTE THE mAP
    p, r, ap, f1, unique_classes = mAP(
        pred_boxes,
        pred_cs,
        pred_ps,
        true_boxes,
        true_classes,
        iou_threshold=iou_threshold
    )
    return p, r, ap, f1, unique_classes

