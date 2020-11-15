import torch
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

S = 7
SS = S*S
B = 2
C = 12

def to_xywh(x1s,y1s,x2s,y2s):
    x_center = torch.mean(torch.cat([x1s,x2s], 1), dim=1)
    y_center = torch.mean(torch.cat([y1s,y2s], 1), dim=1)
    w = torch.sub(x2s, x1s)
    h = torch.sub(y2s, y1s)
    return x_center, y_center, w, h

def determine_grid(x_center,y_center):
    x_grid = torch.floor(x_center * S)
    y_grid = torch.floor(y_center * S)
    grid_num = x_grid*S + y_grid
    return grid_num

# Compute the loss.
def yolo_loss(yolo_output, labels, meta):
    yolo_output_view = yolo_output.view(-1, (B*5+C))
    num_batch = yolo_output_view.shape[0] // (SS)

    H = 224
    W = 224
    boxes = meta["boxes"]
    metadata = meta["metadata"]
    logger.info("len(metadata): {}, len(boxes): {}".format(len(metadata), len(boxes)))

    x_center, y_center, w, h = to_xywh(
        boxes[:,1:2]/W, boxes[:,2:3]/H, boxes[:,3:4]/W, boxes[:,4:5]/H)
    grid = determine_grid(x_center, y_center)

    # break up boxes into batch

    to_batch_num={}
    i_batch = 0
    for _metadata in metadata:
        _metadata = tuple(_metadata.cpu().tolist())
        if _metadata not in to_batch_num:
            to_batch_num[_metadata] = i_batch
            i_batch+=1

    # logger.info("to_batch_num: {}".format(to_batch_num))

    # x_center, y_center, w, h, C, x_center, y_center, w, h, C, p1, p2, p3, .., p12
    yolo_labels = torch.zeros((num_batch, S*S, (B*5+C))).cuda()
    indicator_obj_bbox = torch.zeros((num_batch, S*S, B)).cuda()
    indicator_obj = torch.zeros((num_batch, S*S, 1)).cuda()
    # fill in labels
    # logger.info('boxes.shape: {}'.format(boxes.shape))
    for i_batch in range(num_batch):
        for idx in range(len(boxes)):
            _metadata = tuple(metadata[idx].cpu().tolist())
            i_batch = to_batch_num[_metadata]

            # Set indicator
            indicator_obj_bbox[i_batch, int(grid[idx]), 0] = 1 # Just using the first bouding box, e.g. B=1
            indicator_obj[i_batch, int(grid[idx]), 0] = 1

            # Set bounding box
            yolo_labels[i_batch, int(grid[idx]),0:4] = torch.tensor([
                x_center[idx], y_center[idx], w[idx], h[idx]])

            # Set confidence
            yolo_labels[i_batch, int(grid[idx]), 4] = 1

            # Set class prob
            class_idx = torch.argmax(labels[idx])
            yolo_labels[i_batch, int(grid[idx]),B*5+class_idx] = 1

    # print('preds.shape: {}'.format(preds.shape))
    # print('yolo_output.shape: {}'.format(yolo_output.shape))
    # print('labels.shape: {}'.format(labels.shape))
    # print(meta['boxes'])
    # print(meta['ori_boxes'])



    lambda_coord = 5
    lambda_noobj = 0.5

    yolo_labels = yolo_labels.view(-1, (B*5+C))
    indicator_obj_bbox = indicator_obj_bbox.view(-1, B)
    indicator_obj = indicator_obj.view(-1, 1)

    X = yolo_labels[:,0:-C:5]
    X_hat = yolo_output_view[:,0:-C:5]
    Y = yolo_labels[:,1:-C:5]
    Y_hat = yolo_output_view[:,1:-C:5]
    W = yolo_labels[:,2:-C:5]
    W_hat = torch.exp(yolo_output_view[:,2:-C:5])
    H = yolo_labels[:,3:-C:5]
    H_hat = torch.exp(yolo_output_view[:,3:-C:5]) # make it non-negative
    _C = yolo_labels[:,4:-C:5]
    C_hat = yolo_output_view[:,4:-C:5]
    P = yolo_labels[:,-C:]
    P_hat = yolo_output_view[:,-C:]

    logger.info('W.shape: {}'.format(W.shape))
    logger.info('W_hat.shape: {}'.format(W_hat.shape))
    logger.info('H.shape: {}'.format(H.shape))
    logger.info('H_hat.shape: {}'.format(H_hat.shape))
    loss1 = lambda_coord*indicator_obj_bbox*(torch.square(X-X_hat) + torch.square(Y-Y_hat))
    loss2 = lambda_coord*indicator_obj_bbox*(torch.square(torch.sqrt(W)-torch.sqrt(W_hat)) + torch.square(torch.sqrt(H)-torch.sqrt(H_hat)))
    loss3 = indicator_obj_bbox*torch.square(_C-C_hat)
    loss4 = lambda_noobj*(1-indicator_obj_bbox)*torch.square(_C-C_hat)
    loss5 = indicator_obj*torch.square(P-P_hat)
    return loss1.sum() + loss2.sum() + loss3.sum() + loss4.sum() + loss5.sum()
