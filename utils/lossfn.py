import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LossFunction(nn.Module):
    def __init__(self, S, B, C, lambda_coord, lambda_noobj):
        super(LossFunction, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        # Increase the loss from bounding box coordinate predictions
        # and decrease the loss from cofidence predictions for boxes 
        # that don't contain object
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, box1, box2):

        iou_mask = (box1[:, 0] - box2[:, 1]) >= 0
        iou_mask_1 = (box2[:, 0] - box1[:, 1]) >= 0
        iou_mask_2 = (box2[:, 2] - box1[:, 3]) >= 0
        iou_mask_3 = (box1[:, 2] - box2[:, 3]) >= 0

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

    def forward(self, predictions, ground_truth):
        N = predictions.size()[0]

        # Get the all bounding that confidence score is > 0 and == 0
        # size is [:, :, :]
        coord_mask = ground_truth[:, :, :, 4] > 0
        noobj_mask = ground_truth[:, :, :, 4] == 0

        # Make their size same as the ground_truth by unsqueeze and expand_as
        coord_mask = coord_mask.unsqueeze(-1).expand_as(ground_truth)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(ground_truth)

        # Because each cell has two bounding boxs, and each bounding box
        # predict 5 values, so the top 10 values is the box prediction
        coord_prediction = predictions[coord_mask].view(-1, 23)
        bounding_box_prediction = coord_prediction[:, :10].contiguous().view(-1, 5)
        bounding_box_prediction = F.sigmoid(bounding_box_prediction)
        class_prediction = coord_prediction[:, 10:]
        class_prediction = F.softmax(class_prediction, dim=1)

        coord_gound_truth = ground_truth[coord_mask].view(-1, 23)
        bounding_box_ground_truth = coord_gound_truth[:, :10].contiguous().view(-1, 5)
        class_ground_truth = coord_gound_truth[:, 10:]

        # Compute not contain obj loss which is calculated by confidence
        noobj_prediction = predictions[noobj_mask].view(-1, 23)
        noobj_ground_truth = ground_truth[noobj_mask].view(-1, 23)

        noobj_prediction_confindence_mask = torch.zeros(noobj_prediction.size()).byte()
        noobj_prediction_confindence_mask[:, 4] = 1;
        noobj_prediction_confindence_mask[:, 9] = 1;

        noobj_prediction_confidence = noobj_prediction[noobj_prediction_confindence_mask]
        noobj_ground_truth_confidence = noobj_ground_truth[noobj_prediction_confindence_mask]

        noobj_loss = F.mse_loss(noobj_prediction_confidence.float(), noobj_ground_truth_confidence.float()) * self.lambda_noobj


        # We only want one bounding box predictor to be responsible for each object
        coord_responsible_mask = torch.zeros(bounding_box_prediction.size(0)).byte()
        coord_noresponsible_mask = torch.zeros(bounding_box_prediction.size(0)).byte()
        bounding_box_ground_truth_iou = torch.zeros(bounding_box_ground_truth.size())

        for i in range(0, bounding_box_ground_truth.size(0), 2):
            box1 = bounding_box_prediction[i:i+2].double()
            box2 = bounding_box_ground_truth[i:i+2].double()

            box1_sides = Variable(torch.FloatTensor(box1.size()))
            box2_sides = Variable(torch.FloatTensor(box1.size()))

            box1_sides[:, :2] = box1[:, :2] - 0.5*box1[:, 2:4]
            box1_sides[:, 2:4] = box1[:, :2] + 0.5*box2[:, 2:4]

            box2_sides[:, :2] = box2[:, :2] - 0.5*box1[:, 2:4]
            box2_sides[:, 2:4] = box2[:, :2] + 0.5*box2[:, 2:4]

            # box1 => [x0, y0, x1, y1], box2 => [x0, y0, x1, y1]
            iou = self.compute_iou(box1_sides[:, :4], box2_sides[:, :4])
            responsible_iou_index = torch.argmax(iou, dim=-1)

            coord_responsible_mask[i+responsible_iou_index] = 1
            coord_noresponsible_mask[i+1-responsible_iou_index] = 1
            bounding_box_ground_truth_iou[i+responsible_iou_index] = iou[responsible_iou_index]
        
        bounding_box_prediction_responsible = bounding_box_prediction[coord_responsible_mask].view(-1, 5).float().cuda()
        bounding_box_prediction_noresponsible = bounding_box_prediction[coord_noresponsible_mask].view(-1, 5).float()

        bounding_box_ground_truth_responsible_iou = bounding_box_ground_truth_iou[coord_responsible_mask].view(-1, 5).float().cuda()
        bounding_box_ground_truth_responsible = bounding_box_ground_truth[coord_responsible_mask].view(-1, 5).float()

        print(bounding_box_ground_truth_responsible_iou.mean())
        responsible_confidence_loss = F.mse_loss(bounding_box_prediction_responsible[:, 4], bounding_box_ground_truth_responsible_iou[:, 4])
        responsible_xy_loss = F.mse_loss(bounding_box_prediction_responsible[:, :2], bounding_box_ground_truth_responsible[:, :2])
        responsible_wh_loss = F.mse_loss(bounding_box_prediction_responsible[:, 2:4], bounding_box_ground_truth_responsible[:, 2:4])

        # class loss
        class_loss = F.mse_loss(class_prediction.float(), class_ground_truth.float())

        total_loss = self.lambda_coord*(responsible_xy_loss + responsible_wh_loss) + responsible_confidence_loss + self.lambda_noobj*noobj_loss + class_loss
        return total_loss


if __name__ == "__main__":
    ls = LossFunction(1, 1, 2, 5, 1)
    pre_data = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]]).view(1, 2, 2, 4)
    gt_data = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]]).view(1, 3, 3, 4)
    print(ls.forward(pre_data, gt_data))

