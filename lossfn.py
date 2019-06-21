import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LossFunction(nn.Module):
    def __init__(self, 
                S, 
                B, 
                C, 
                object_scale, 
                no_object_scale,
                coord_scale,
                class_scale
            ):
        super(LossFunction, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.C_predict = self.B*5+self.C
        self.B_predict = self.B*5

        # Increase the loss from bounding box coordinate predicts
        # and decrease the loss from cofidence predicts for boxes # that don't contain object
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

    def compute_iou(self, box1, box2):

        iou_mask = (box1[:, 0] - box2[:, 2]) >= 0
        iou_mask_1 = (box2[:, 0] - box1[:, 2]) >= 0
        iou_mask_2 = (box2[:, 1] - box1[:, 3]) >= 0
        iou_mask_3 = (box1[:, 1] - box2[:, 3]) >= 0

        box1_w = box1[:, 2] - box1[:, 0]
        box1_h = box1[:, 3] - box1[:, 1]

        box2_w = box2[:, 2] - box2[:, 0]
        box2_h = box2[:, 3] - box2[:, 1]

        box1_area = box1_w * box1_h
        box2_area = box2_w * box2_h

        inter_box_x0 = torch.max(box1[:, 0], box2[:, 0])
        inter_box_y0 = torch.max(box1[:, 1], box2[:, 1])
        inter_box_x1 = torch.min(box1[:, 2], box2[:, 2])
        inter_box_y1 = torch.min(box1[:, 3], box2[:, 3])

        inter_box_w = inter_box_x1 - inter_box_x0
        inter_box_h = inter_box_y1 - inter_box_y0

        inter = inter_box_w * inter_box_h
        iou = inter/ (box1_area + box2_area - inter + 1e-6)

        iou[iou_mask] = 0
        iou[iou_mask_1] = 0
        iou[iou_mask_2] = 0
        iou[iou_mask_3] = 0

        return iou

    def _gen_mask(self, target):

        # Get the all bounding that confidence score is > 0 and == 0
        # size is [:, :, :]
        mask_target = target[:, :, :, :self.B_predict].view(-1, self.S, self.S, self.B, self.B)
        mask_target = mask_target[:, :, :, :, 4].sum(dim=-1)
        coord_mask = mask_target[:, :, :] > 0
        noobj_mask = mask_target[:, :, :] == 0

        # Make their size same as the target by unsqueeze and expand_as
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        return coord_mask, noobj_mask

        
    def forward(self, predicts, target, anchor_box):
        anchor_box = torch.FloatTensor(anchor_box).cuda()
        N = predicts.size()[0]

        coord_mask, noobj_mask = self._gen_mask(target)

        # Because each cell has two bounding boxs, and each bounding box
        # predict 5 values, so the top 10 values is the box predict
        coord_predict = predicts[coord_mask].view(-1, self.C_predict)
        bnd_predict = coord_predict[:, :self.B_predict].contiguous().view(-1, self.B, 5)
        bnd_predict[:, :, :2] = (bnd_predict[:, :, :2]).sigmoid()
        bnd_predict[:, :, 2:4]  = (bnd_predict[:, :, 2:4]).tanh().exp() * anchor_box
        bnd_predict[:, :, 4] = (bnd_predict[:, :, 4]).sigmoid()

        class_predict = coord_predict[:, self.B_predict:]
        class_predict = F.softmax(class_predict, dim=1)

        coord_gound_truth = target[coord_mask].view(-1, self.C_predict)
        bnd_target = coord_gound_truth[:, :self.B_predict].contiguous().view(-1, self.B, 5)
        class_target = coord_gound_truth[:, self.B_predict:]

        # Compute not contain obj loss which is calculated by confidence
        noobj_predict = predicts[noobj_mask].view(-1, self.C_predict)
        noobj_target = target[noobj_mask].view(-1, self.C_predict)

        noobj_predict_confindence_mask = torch.zeros(noobj_predict.size()).byte()

        for b in range(self.B):
            noobj_predict_confindence_mask[:, 4+5*b] = 1;

        noobj_predict_confidence = noobj_predict[noobj_predict_confindence_mask]
        noobj_predict_confidence = noobj_predict_confidence.sigmoid()
        noobj_target_confidence = noobj_target[noobj_predict_confindence_mask]

        # We only want one bounding box predictor to be respon for each object
        coord_respon_mask = torch.zeros(bnd_predict.size(0), bnd_predict.size(1)).byte()
        coord_norespon_mask = torch.zeros(bnd_predict.size(0), bnd_predict.size(1)).byte()
        bnd_target_iou = torch.zeros(bnd_target.size())


        for i in range(0, bnd_target.size(0)):
            box1 = bnd_predict[i].double()
            box2 = bnd_target[i].double()

            box1_sides = Variable(torch.FloatTensor(box1.size()))
            box2_sides = Variable(torch.FloatTensor(box1.size()))

            box1_sides[:, :2] = box1[:, :2]*32 - 0.5*box1[:, 2:4]*448
            box1_sides[:, 2:4] = box1[:, :2]*32 + 0.5*box1[:, 2:4]*448

            box2_sides[:, :2] = box2[:, :2]*32 - 0.5*box2[:, 2:4]*448
            box2_sides[:, 2:4] = box2[:, :2]*32 + 0.5*box2[:, 2:4]*448

            # box1 => [x0, y0, x1, y1], box2 => [x0, y0, x1, y1]
            iou = self.compute_iou(box1_sides[:, :4], box2_sides[:, :4])
            respon_iou_index = torch.argmax(box2[:, 4])
            
            if box2[respon_iou_index, 4] == 1:
                coord_respon_mask[i, respon_iou_index] = 1
                bnd_target_iou[i, respon_iou_index] = iou[respon_iou_index]
    
        bnd_predict_respon = bnd_predict[coord_respon_mask].view(-1, 5).float().cuda()

        bnd_target_respon_iou = bnd_target_iou[coord_respon_mask].view(-1, 5).float().cuda()
        bnd_target_respon = bnd_target[coord_respon_mask].view(-1, 5).float().cuda()

        confidence_loss = F.mse_loss(bnd_predict_respon[:, 4], bnd_target_respon_iou[:, 4], size_average = False)
        noobj_loss = F.mse_loss(noobj_predict_confidence.float(), noobj_target_confidence.float(), size_average = False)
        xy_loss = F.mse_loss(bnd_predict_respon[:, :2], bnd_target_respon[:, :2], size_average = False)
        wh_loss = F.mse_loss(bnd_predict_respon[:, 2:4], bnd_target_respon[:, 2:4], size_average = False)
        class_loss = F.mse_loss(class_predict.float(), class_target.float(), size_average = False)
        total_loss = self.coord_scale*(xy_loss + wh_loss) + self.object_scale*confidence_loss + self.no_object_scale*noobj_loss + self.class_scale*class_loss

        return total_loss/N

