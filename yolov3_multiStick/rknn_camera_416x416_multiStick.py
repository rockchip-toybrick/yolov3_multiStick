import logging as log
import time
import numpy as np
import cv2
import sys
import os
import signal
from rknn.api import RKNN
from multiprocessing import Process, Queue, Lock
import multiprocessing

GRID0 = 13
GRID1 = 26
GRID2 = 52
LISTSIZE = 85
SPAN = 3
NUM_CLS = 80
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop        ","mouse        ","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    obj_thresh = -np.log(1/OBJ_THRESH - 1)
    pos = np.where(box_confidence > obj_thresh)
    input = input[pos]
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    for idx, val in enumerate(pos[2]):
        box_wh[idx] = box_wh[idx] * anchors[pos[2][idx]]
    pos0 = np.array(pos[0])[:, np.newaxis]
    pos1 = np.array(pos[1])[:, np.newaxis]
    grid = np.concatenate((pos1, pos0), axis=1)
    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):
    # # yolov3
    # masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    #            [59, 119], [116, 90], [156, 198], [373, 326]]
    # yolov3-tiny
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # # Scale boxes back to original image shape.
    # width, height = 416, 416 #shape[1], shape[0]
    # image_dims = [width, height, width, height]
    # boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # print('class: {0}, score: {1:.2f}'.format(CLASSES[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))

def load_model(dev_idx):
        global threadSem
        
        rknn = RKNN()
        print('-->loading model')
        rknn.load_rknn('./yolov3_416x416.rknn')
        #rknn.load_rknn('./yolov3.rknn')
        print('loading model done')

        print('--> Init runtime environment')

        _, ntb_devices = rknn.list_devices()
        if len(ntb_devices) > 0 and int(dev_idx) < len(ntb_devices):
            dev_id = ntb_devices[int(dev_idx)]
        else:
            dev_id = ntb_devices[0]
            print('W No device left.')
        ret = rknn.init_runtime(target='rk1808', device_id=dev_id)
        if ret != 0:
                print('Init runtime environment failed')
                exit(ret)
        print('done')
        print("dev_id=%s" % (dev_id))
        return rknn


def video_capture(q_frame:Queue, q_image:Queue, flag):
    video = cv2.VideoCapture(0)
    print("video.isOpened()={}", video.isOpened())
    try:
        while True:
            if flag.value == 20:
                if video.isOpened():
                    video.release()
                    print("video release!")
                print("exit video_capture!")
                break
            s = time.time()
            #print('capture q_image.qsize() = {}. '.format(q_image.qsize()))
            ret, frame = video.read()
            assert ret, 'read video frame failed.'
            #print('capture read used {} ms.'.format((time.time() - s) * 1000))

            s = time.time()
            image = cv2.resize(frame, (416, 416))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print('capture resize used {} ms.'.format((time.time() - s) * 1000))

            s = time.time()
            if q_frame.empty():
                q_frame.put(frame)
            if q_image.full():
                continue
            else:
                q_image.put(image)
            #print("capture put to queue used {} ms".format((time.time()-s)*1000))
    except KeyboardInterrupt:
        video.release()
        print("exit video_capture!")

def infer_rknn(dev_idx, q_image:Queue, q_infer:Queue, flag):
    rknn = load_model(dev_idx)
    try:
        while True:
            if flag.value == 10:
                print("befor exit infer rknn")
                rknn.release()
                print("exit infer_rknn!")
                flag.value = 20
                break
            s = time.time()
            if q_image.empty():
                continue
            else:
                image = q_image.get()
            #print('Infer{} q_image.qsize() = {}. '.format(dev_idx, q_image.qsize()))
            #print('Infer get, used time {} ms. '.format((time.time() - s) * 1000))

            s = time.time()
            out_boxes, out_boxes2, out_boxes3 = rknn.inference(inputs=[image])
            out_boxes = out_boxes.reshape(SPAN, LISTSIZE, GRID0, GRID0)
            out_boxes2 = out_boxes2.reshape(SPAN, LISTSIZE, GRID1, GRID1)
            out_boxes3 = out_boxes3.reshape(SPAN, LISTSIZE, GRID2, GRID2)
            input_data = []
            input_data.append(np.transpose(out_boxes, (2, 3, 0, 1)))
            input_data.append(np.transpose(out_boxes2, (2, 3, 0, 1)))
            input_data.append(np.transpose(out_boxes3, (2, 3, 0, 1)))
            #print('Infer done, used time {} ms. '.format((time.time() - s) * 1000))

            s = time.time()
            if q_infer.full():
                continue
            else:
                q_infer.put(input_data)
            #print('Infer put, used time {} ms. '.format((time.time() - s) * 1000))
    except KeyboardInterrupt:
        print("befor exit infer rknn")
        rknn.release()
        print("exit infer_rknn!")

def post_process(q_infer, q_objs, flag):
    while True:
        if flag.value == 20:
            break
        s = time.time()
        if q_infer.empty():
            continue
        else:
            input_data = q_infer.get()
        #print('Post process q_infer.qsize() = {}. '.format(q_infer.qsize()))
        #print('Post process get, used time {} ms. '.format((time.time() - s) * 1000))

        s = time.time()
        boxes, classes, scores = yolov3_post_process(input_data)
        #print('Post process done, used time {} ms. '.format((time.time() - s) * 1000))

        s = time.time()
        if q_objs.full():
            continue
        else:
            q_objs.put((boxes, classes, scores))
        #print('Post process put, used time {} ms. '.format((time.time()-s)*1000))


if __name__ == '__main__':
    #log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG)

    q_frame = Queue(maxsize=1)
    q_image = Queue(maxsize=12)
    q_infer = Queue(maxsize=6)
    q_objs = Queue(maxsize=6)
    flag = multiprocessing.Value("d", 0)

    p_cap1 = Process(target=video_capture, args=(q_frame, q_image, flag))
    #p_cap2 = Process(target=video_capture, args=(q_frame, q_image, flag))
    p_infer1 = Process(target=infer_rknn, args=(0, q_image, q_infer, flag))
    p_infer2 = Process(target=infer_rknn, args=(1, q_image, q_infer, flag))
    #p_infer3 = Process(target=infer_rknn, args=(2, q_image, q_infer, flag))
    #p_infer4 = Process(target=infer_rknn, args=(3, q_image, q_infer, flag))
    #p_infer5 = Process(target=infer_rknn, args=(4, q_image, q_infer, flag))
    #p_infer6 = Process(target=infer_rknn, args=(5, q_image, q_infer, flag))
    p_post1 = Process(target=post_process, args=(q_infer, q_objs, flag))
    p_post2 = Process(target=post_process, args=(q_infer, q_objs, flag))


    p_cap1.start()
    #p_cap2.start()
    p_infer1.start()
    p_infer2.start()
    #p_infer3.start()
    #p_infer4.start()
    #p_infer5.start()
    #p_infer6.start()
    p_post1.start()
    p_post2.start()

    fps = 0
    l_used_time = []

    try:
        while True:
            s = time.time()
            #print('main func, q_frame.qsize() = {}. '.format(q_frame.qsize()))
            frame = q_frame.get()
            boxes, classes, scores = q_objs.get()
            #print('main func, get objs use {} ms. '.format((time.time() - s) * 1000))

            if boxes is not None:
                draw(frame, boxes, scores, classes)
            cv2.putText(frame, text='FPS: {}'.format(fps), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.imshow("results", frame)

            c = cv2.waitKey(5) & 0xff
            if c == 27:
                flag.value = 10
                time.sleep(5)
                cv2.destroyAllWindows()
                print("ESC, exit main!")
                break

            used_time = time.time() - s
            l_used_time.append(used_time)
            if len(l_used_time) > 20:
                l_used_time.pop(0)
            fps = int(1/np.mean(l_used_time))
            #print('main func, used time {} ms. '.format(used_time*1000))
    except KeyboardInterrupt:
        time.sleep(5)
        cv2.destroyAllWindows()
        print("ctrl + c, exit main!")

    p_cap1.terminate()
    #p_cap2.terminate()
    p_infer1.terminate()
    p_infer2.terminate()
    #p_infer3.terminate()
    #p_infer4.terminate()
    #p_infer5.terminate()
    #p_infer6.terminate()
    p_post1.terminate()
    p_post2.terminate()
    sys.exit()
