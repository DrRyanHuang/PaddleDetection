# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import paddle
from argparse import Namespace
import numpy as np


class TaskCategory():
    def __init__(self, task_file, num_classes):
        task_content = json.load(open(task_file))
        self.tasks = [Namespace(
            index=idx,
            name=it['name'],
            num_steps=it['num_steps'],
            required_outputs=it['required_outputs'],
            required_targets=it['required_targets'],
            losses=it['losses'],
        ) for idx, it in enumerate(task_content)]
        self.name2task = {
            t.name: t for t in self.tasks
        }

        num_cats = [it['num_cats'] for it in task_content] # [80]
        currentIndex = 0
        all_cats = num_cats[0] # 80
        self.id_to_index = []
        for i in range(num_classes):
            if i >= all_cats:
                currentIndex += 1
                all_cats += num_cats[currentIndex]
            self.id_to_index.append(currentIndex)
        # self.id_to_index = paddle.LongTensor(self.id_to_index)
        self.id_to_index = paddle.to_tensor(self.id_to_index, dtype="int64", place=paddle.CPUPlace()) # why all zero?

    def __getitem__(self, tId):
        if isinstance(tId, int):
            return self.tasks[tId]
        elif isinstance(tId, str):
            return self.name2task[tId]
        else:
            return NotImplemented

    def getTaskCorrespondingIds(self, bs_idx, cls_idx):
        """
        Args:
            cls_idx: Tensor(bs, cs)
        """
        taskIndexes = self.id_to_index[cls_idx] # cs_all
        tasks = {}
        for taskIdx in taskIndexes.unique(): # 只有一个任务
            tasks[taskIdx.item()] = {
                "indexes": (taskIndexes == taskIdx),
                "cls_idx": cls_idx[taskIndexes == taskIdx],
                "bs_idx": bs_idx[taskIndexes == taskIdx],
            }
        return tasks

    def arrangeBySteps(self, cls_idx, *args):
        tIds = [self.id_to_index[ic] for ic in cls_idx]
        nSteps = paddle.to_tensor([self.tasks[tId].num_steps for tId in tIds], place=paddle.CPUPlace())
        
        # 这里排序排个毛线?
        nSteps, indices = nSteps.sort(descending=True).cpu(), nSteps.argsort(descending=True).cpu()
        
        return (nSteps, cls_idx[indices], *[a[indices] if a is not None else None for a in args])

    def getNumSteps(self, cls_idx, *args):
        tIds = [self.id_to_index[ic] for ic in cls_idx]
        nSteps = paddle.to_tensor([self.tasks[tId].num_steps for tId in tIds], place=paddle.CPUPlace())
        return nSteps