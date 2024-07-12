import numpy as np
import torch
import torch.nn as nn
from utils.modules import Conv, reorg_layer

from backbone import build_backbone
import tools


class YOLOv2D19(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, anchor_size=None):
        super(YOLOv2D19, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

        # backbone darknet-19
        self.backbone = build_backbone(model_name='darknet19', pretrained=trainable)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)
        
        # prediction layer
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)


    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                xywh_pred : [B, H*W*anchor_n, 4] \n
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * self.all_anchor_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * self.stride

        return xywh_pred
    

    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @ torch.no_grad()
    def inference(self, x):
        # backbone
        feats = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        B, abC, H, W = pred.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # [B, H*W*num_anchor, 1]
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        # decode box
        reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)
        box_pred = self.decode_boxes(reg_pred)

        # batch size = 1
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # score
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # normalize bbox
        bboxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            feats = self.backbone(x)

            # reorg layer
            p5 = self.convsets_1(feats['layer3'])
            p4 = self.reorg(self.route_layer(feats['layer2']))
            p5 = torch.cat([p4, p5], dim=1)

            # head
            p5 = self.convsets_2(p5)

            # pred
            pred = self.pred(p5)

            B, abC, H, W = pred.size()

            # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

            # [B, H*W*num_anchor, 1]
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
            # [B, H*W, num_anchor, num_cls]
            cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
            # [B, H*W, num_anchor, 4]
            reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
            reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)

            # decode bbox
            x1y1x2y2_pred = (self.decode_boxes(reg_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            reg_pred = reg_pred.view(B, H*W*self.num_anchors, 4)

            # set conf target
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
            gt_conf = iou_pred.clone().detach()

            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

            # loss
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                iou_loss
            ) = tools.loss(pred_conf=conf_pred,
                           pred_cls=cls_pred,
                           pred_txtytwth=reg_pred,
                           pred_iou=iou_pred,
                           label=target
                           )

            return conf_loss, cls_loss, bbox_loss, iou_loss   




class YOLOv2D19_QAT(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, anchor_size=None):
        super(YOLOv2D19_QAT, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

        # backbone darknet-19
        self.backbone = build_backbone(model_name='darknet19', pretrained=trainable)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)
        
        # prediction layer
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)


    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                xywh_pred : [B, H*W*anchor_n, 4] \n
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * self.all_anchor_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * self.stride

        return xywh_pred
    

    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @ torch.no_grad()
    def inference(self, x):
        # backbone
        feats = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        B, abC, H, W = pred.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # [B, H*W*num_anchor, 1]
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        # decode box
        reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)
        box_pred = self.decode_boxes(reg_pred)

        # batch size = 1
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # score
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # normalize bbox
        bboxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        # if not self.trainable:
        #     return self.inference(x)
        # else:
        # backbone
        feats = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        return pred



def eval_output(pred, input_size):
    
    anchor_size = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    anchor_size = torch.tensor(anchor_size)
    num_anchors = len(anchor_size)
    num_classes = 20
    # input_size = 416

    B, abC, H, W = pred.size()

    # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
    pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

    # [B, H*W*num_anchor, 1]
    conf_pred = pred[:, :, :1 * num_anchors].contiguous().view(B, H*W*num_anchors, 1)
    # [B, H*W, num_anchor, num_cls]
    cls_pred = pred[:, :, 1 * num_anchors : (1 + num_classes) * num_anchors].contiguous().view(B, H*W*num_anchors, num_classes)
    # [B, H*W, num_anchor, 4]
    reg_pred = pred[:, :, (1 + num_classes) * num_anchors:].contiguous()
    # decode box
    reg_pred = reg_pred.view(B, H*W, num_anchors, 4)
    box_pred = decode_boxes(reg_pred, input_size)

    # batch size = 1
    conf_pred = conf_pred[0]
    cls_pred = cls_pred[0]
    box_pred = box_pred[0]

    # score
    scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

    # normalize bbox
    bboxes = torch.clamp(box_pred / input_size, 0., 1.)

    # to cpu
    scores = scores.to('cpu').detach().numpy()
    bboxes = bboxes.to('cpu').detach().numpy()

    # post-process
    bboxes, scores, cls_inds = postprocess(bboxes, scores)

    return bboxes, scores, cls_inds





def postprocess(bboxes, scores):
    """
    bboxes: (HxW, 4), bsize = 1
    scores: (HxW, num_classes), bsize = 1
    """
    num_classes=20
    conf_thresh=0.001
    
    cls_inds = np.argmax(scores, axis=1)
    scores = scores[(np.arange(scores.shape[0]), cls_inds)]
    
    # threshold
    keep = np.where(scores >= conf_thresh)
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # NMS
    keep = np.zeros(len(bboxes), dtype=int)
    for i in range(num_classes):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    bboxes = bboxes[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    return bboxes, scores, cls_inds


def nms(dets, scores):
    """"Pure Python NMS baseline."""
    nms_thresh=0.5
    x1 = dets[:, 0]  #xmin
    y1 = dets[:, 1]  #ymin
    x2 = dets[:, 2]  #xmax
    y2 = dets[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def create_grid(input_size):
    
    anchor_size = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    anchor_size = torch.tensor(anchor_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = 32
    w, h = input_size, input_size
    # generate grid cells
    ws, hs = w // stride, h // stride
    grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(device)

    # generate anchor_wh tensor
    anchor_wh = anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(device)

    return grid_xy, anchor_wh



def decode_xywh(txtytwth_pred, input_size):
    """
        Input: \n
            txtytwth_pred : [B, H*W, anchor_n, 4] \n
        Output: \n
            xywh_pred : [B, H*W*anchor_n, 4] \n
    """
    # input_size = 416
    grid_cell, all_anchor_wh = create_grid(input_size)
    stride = 32
    B, HW, ab_n, _ = txtytwth_pred.size()
    # b_x = sigmoid(tx) + gride_x
    # b_y = sigmoid(ty) + gride_y
    # grid_cell.to(txtytwth_pred.device)
    # all_anchor_wh.to(txtytwth_pred.device)
    xy_pred = torch.sigmoid(txtytwth_pred[..., :2]).to(grid_cell.device) + grid_cell
    # b_w = anchor_w * exp(tw)
    # b_h = anchor_h * exp(th)
    wh_pred = torch.exp(txtytwth_pred[..., 2:]).to(all_anchor_wh.device) * all_anchor_wh
    # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
    xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * stride

    return xywh_pred

def decode_boxes(txtytwth_pred, input_size):
    """
        Input: \n
            txtytwth_pred : [B, H*W, anchor_n, 4] \n
        Output: \n
            x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
    """
    # txtytwth -> cxcywh
    xywh_pred = decode_xywh(txtytwth_pred, input_size)

    # cxcywh -> x1y1x2y2
    x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
    x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
    x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
    
    return x1y1x2y2_pred

def train_output(pred, target, input_size):

    anchor_size = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    anchor_size = torch.tensor(anchor_size)
    num_anchors = len(anchor_size)
    num_classes = 20
    # input_size = 416

    B, abC, H, W = pred.size()

    # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
    pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

    # [B, H*W*num_anchor, 1]
    conf_pred = pred[:, :, :1 * num_anchors].contiguous().view(B, H*W*num_anchors, 1)
    # [B, H*W, num_anchor, num_cls]
    cls_pred = pred[:, :, 1 * num_anchors : (1 + num_classes) * num_anchors].contiguous().view(B, H*W*num_anchors, num_classes)
    # [B, H*W, num_anchor, 4]
    reg_pred = pred[:, :, (1 + num_classes) * num_anchors:].contiguous()
    reg_pred = reg_pred.view(B, H*W, num_anchors, 4)

    # decode bbox
    x1y1x2y2_pred = (decode_boxes(reg_pred, input_size) / input_size).view(-1, 4)
    x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
    # print(x1y1x2y2_pred.device)
    # print(x1y1x2y2_gt.device)
    x1y1x2y2_pred = x1y1x2y2_pred.to('cpu')
    # print(x1y1x2y2_pred.device)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # x1y1x2y2_gt = x1y1x2y2_gt.to(device)

    reg_pred = reg_pred.view(B, H*W*num_anchors, 4)

    # set conf target
    iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
    gt_conf = iou_pred.clone().detach()

    # gt_conf = gt_conf.to(target.device)


    # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
    target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

    # print(conf_pred.device)
    # print(cls_pred.device)
    # print(reg_pred.device)
    # print(iou_pred.device)
    # print(target.device)


    # loss
    (
        conf_loss,
        cls_loss,
        bbox_loss,
        iou_loss
    ) = tools.loss(pred_conf=conf_pred,
                    pred_cls=cls_pred,
                    pred_txtytwth=reg_pred,
                    pred_iou=iou_pred,
                    label=target
                    )

    return conf_loss, cls_loss, bbox_loss, iou_loss 


class QuantQAT_YOLOv2D19(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, anchor_size=None, scales = [1,1,1,1,1,1,1]):
        super(QuantQAT_YOLOv2D19, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

        # backbone darknet-19
        self.backbone = build_backbone(model_name='darknet19', pretrained=trainable)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)
        
        # prediction layer
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

        # self.scales = [1,1,1,1,1,1,1]
        #scales = [output scale layer1, out_scale layer2, out_scale layer3, out_p4, out_p5, out_cat, final_scale/out_pred]
        self.scales = scales
        

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                xywh_pred : [B, H*W*anchor_n, 4] \n
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * self.all_anchor_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * self.stride

        return xywh_pred
    

    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @ torch.no_grad()
    def inference(self, x):
        # backbone
        feats = self.backbone(x)


        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = quantcat(p4,p5, self.scales[3], self.scales[4], self.scales[5])
        # p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        pred = pred * self.scales[6]

        B, abC, H, W = pred.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # [B, H*W*num_anchor, 1]
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        # decode box
        reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)
        box_pred = self.decode_boxes(reg_pred)

        # batch size = 1
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # score
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # normalize bbox
        bboxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):

        # backbone
        feats = self.backbone(x)

        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = quantcat(p4,p5, self.scales[3], self.scales[4], self.scales[5])

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        return pred  


class QuantYOLOv2D19(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, anchor_size=None, scales = [1,1,1,1,1,1,1]):
        super(QuantYOLOv2D19, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

        # backbone darknet-19
        self.backbone = build_backbone(model_name='darknet19', pretrained=trainable)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)
        
        # prediction layer
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

        # self.scales = [1,1,1,1,1,1,1]
        #scales = [output scale layer1, out_scale layer2, out_scale layer3, out_p4, out_p5, out_cat, final_scale/out_pred]
        self.scales = scales
        

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                xywh_pred : [B, H*W*anchor_n, 4] \n
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * self.all_anchor_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * self.stride

        return xywh_pred
    

    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @ torch.no_grad()
    def inference(self, x):
        # backbone
        feats = self.backbone(x)


        # reorg layer
        p5 = self.convsets_1(feats['layer3'])
        p4 = self.reorg(self.route_layer(feats['layer2']))
        p5 = quantcat(p4,p5, self.scales[3], self.scales[4], self.scales[5])
        # p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # pred
        pred = self.pred(p5)

        pred = pred * self.scales[6]

        B, abC, H, W = pred.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # [B, H*W*num_anchor, 1]
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        # decode box
        reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)
        box_pred = self.decode_boxes(reg_pred)

        # batch size = 1
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # score
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # normalize bbox
        bboxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            feats = self.backbone(x)

            # reorg layer
            p5 = self.convsets_1(feats['layer3'])
            p4 = self.reorg(self.route_layer(feats['layer2']))
            p5 = torch.cat([p4, p5], dim=1)

            # head
            p5 = self.convsets_2(p5)

            # pred
            pred = self.pred(p5)

            B, abC, H, W = pred.size()

            # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

            # [B, H*W*num_anchor, 1]
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
            # [B, H*W, num_anchor, num_cls]
            cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
            # [B, H*W, num_anchor, 4]
            reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
            reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)

            # decode bbox
            x1y1x2y2_pred = (self.decode_boxes(reg_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            reg_pred = reg_pred.view(B, H*W*self.num_anchors, 4)

            # set conf target
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
            gt_conf = iou_pred.clone().detach()

            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

            # loss
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                iou_loss
            ) = tools.loss(pred_conf=conf_pred,
                           pred_cls=cls_pred,
                           pred_txtytwth=reg_pred,
                           pred_iou=iou_pred,
                           label=target
                           )

            return conf_loss, cls_loss, bbox_loss, iou_loss   


def quantcat(p4, p5, p4_out_scale, p5_out_scale, cat_out_scale):
    p4 = p4*p4_out_scale
    p5 = p5*p5_out_scale
    p5 = torch.cat([p4, p5], dim=1)

    p5 = torch.round(p5/cat_out_scale)
    p5 = p5.clamp(max=127)
    p5 = p5.clamp(min=-128)

    return p5
    