# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# from util.misc import inverse_sigmoid
from .task_category import TaskCategory



def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1/x2)





class DetPoseProcess(nn.Layer):
    def __init__(self, args):
        super(DetPoseProcess, self).__init__()
        self.taskCategory = TaskCategory(args.task_category, args.num_classes)
        self.postprocessors = {
            "detection": self.post_process_detection,
            "pose": self.post_process_pose,
        }
        #return NotImplemented

    def forward(self, signals, pred_logits, reference_points, bs_idx, cls_idx):
        """Rearrange results as outputs
        
        Args:
            signals: [Tensor( cs_all, nobj)]:
            reference_points: Tensor( cs_all, nobj, 2 )
            pred_logits: Tensor( cs_all, nboj )
            cls_idx: Tensor( cs_all )
        """
        cs_all, nobj = pred_logits.shape
        # if cls_idx is None:
        #     # assert cs == 1, "Only 1 class is supported without class indexes."
        #     bs_idx = paddle.arange(bs)
        #     cls_idx = paddle.zeros(bs, dtype=paddle.long)
        taskInfos = self.rearrange_by_task(signals, reference_points, bs_idx, cls_idx)
        outputs = {}
        for tId in taskInfos:
            tName = self.taskCategory.tasks[tId].name
            taskInfo = taskInfos[tId]
            output_per_task = self.postprocessors[tName](taskInfo["feats"], 
                                                         taskInfo["reference_points"])
            output_per_task.update({
                "pred_logits": pred_logits[taskInfo["indexes"]],
                "class_index": taskInfo["cls_idx"],
                "batch_index": taskInfo["bs_idx"],
            })
            outputs[tName] = output_per_task
        return outputs

    def rearrange_by_task(self, signals, reference_points, bs_idx, cls_idx):
        taskInfos = self.taskCategory.getTaskCorrespondingIds(bs_idx, cls_idx)
        for tId, taskInfo in taskInfos.items():
            steps = self.taskCategory.tasks[tId].num_steps
            taskInfo["feats"] = paddle.stack(
                [sgn[taskInfo["indexes"][:sgn.shape[0]]] for sgn in signals[:steps]],
                axis=-1
            )
            taskInfo["reference_points"] = reference_points[taskInfo["indexes"]]
        return taskInfos

    def update_outputs(self, outputs, output_per_task, indexes, taskIdx):
        for key in self.taskCategory[taskIdx].required_outputs:
            if key == "pred_logits":
                continue
            elif key in outputs:
                outputs[key] = paddle.concat([outputs[key], output_per_task[key]], axis=1)
            else:
                outputs[key] = output_per_task[key]
        return outputs

    # 一般目标检测后处理
    def post_process_detection(self, signals, reference_points):
        reference = inverse_sigmoid(reference_points) # bs, cs, nobj, 2
        signals[..., :reference.shape[-1]] += reference
        # boxes = signals[..., :4].sigmoid()
        boxes = F.sigmoid(signals[..., :4])
        return {"pred_boxes": boxes}

    # pose检测后处理
    def post_process_pose(self, signals, reference_points):
        cs_all, nobj, _ = reference_points.shape
        reference = inverse_sigmoid(reference_points) # bs, cs, nobj, 2
        signals[..., :2] += reference
        # boxes = signals[..., :4].sigmoid()
        boxes = F.sigmoid(signals[..., :4])
        kpt_out = signals[..., 4:38].reshape([cs_all, nobj, 17, 2])
        outputs_keypoint = boxes[..., None, :2] + kpt_out * boxes[..., None, 2:]
        return {
            "pred_boxes": boxes,
            "pred_keypoints": outputs_keypoint,
        }


# def build_sequence_postprocess(args):
#     return DetPoseProcess(args)
    
