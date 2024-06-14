import os
import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

class my_Annotator(Annotator):
    def __init__(self, im, line_width=None, font_size=0.5, **kwargs):
        super().__init__(im, line_width, **kwargs)
        self.line_width = line_width
        self.font_size = font_size

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        lw = self.lw if self.line_width is None else self.line_width  # line width
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        if label:
            tf = max(lw - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=self.font_size, thickness=tf)[0]
            p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2), 0, self.font_size, txt_color, thickness=tf, lineType=cv2.LINE_AA)

def detect_and_save_video(weights, source, output, img_size=640, conf_thres=0.25, iou_thres=0.45, device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    img_size = check_img_size(img_size, s=stride)

    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=pt)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = None

    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        for i, det in enumerate(pred):
            im0 = im0s.copy()
            annotator = my_Annotator(im0, line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            # 저장
            if vid_writer is None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else im0.shape[1]
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
                vid_writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

            vid_writer.write(im0)

    vid_writer.release()
    print(f'Results saved to {output}')

if __name__ == '__main__':
    weights = 'runs/train/yolov5n8/weights/best.pt'
    source = 'datasets/nuscenes/val/images'
    output = 'output_video.mp4'

    detect_and_save_video(weights, source, output)
