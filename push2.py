# coding=gbk
import cv2
import numpy as np
import yaml
from openvino.inference_engine import IECore
from openvino.runtime import Core  # the version of openvino >= 2022.1
import random
import os
import time
import subprocess as sp

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), scaleup=False, stride=32):
    """
    ��ͼƬ���ŵ�����ָ����С,1920x1080��ͼƬ���ջ����ŵ�640x384�Ĵ�С����YOLOv4��letterbox��һ��
    Resize and pad image while meeting stride-multiple constraints
    :param img: ԭͼ hwc
    :param new_shape: ���ź����ߴ�С
    :param color: pad����ɫ
    :param auto: True�����о������  False��ֱ�ӽ���resize
    :param scale_up: True���Խ����ϲ��� False���������ϲ���
    :return: img: letterbox���ͼƬ HWC
             ratio: wh ratios
             (dw, dh): w��h��pad
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # ֻ�����²��� ��Ϊ�ϲ�������ͼƬģ��
    # (for better test mAP) scale_up = False ���ڴ���new_shape��r<1����ԭͼ��������,С��new_shape��r>1���Ĳ���
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # �����ȡ��������Ա�֤padding���ͼƬ��32��������(416x416)�������(512x512)���Ա�֤��64��������
    # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # �ڽ�С�ߵ��������pad, ��������һ��pad
    # divide padding into 2 sides
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img,ratio,(dw,dh)

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:,0], b2[:,1], b2[:,2], b2[:,3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou

#�Ǽ���ֵ���ƺ���
def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.4, ratio=1, pad=(20,20)):
    # ȡ��batch_size
    bs = np.shape(boxes)[0]
    # xywh___ to____ xyxy
    shape_boxes = np.zeros_like(boxes[:,:,:4])
    shape_boxes[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
    shape_boxes[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
    shape_boxes[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
    shape_boxes[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2
    boxes[:, :, :4] = shape_boxes
    boxes[:, :, 5:] *= boxes[:, :, 4:5]

    # output���ÿһ��ͼƬ��Ԥ����������׶�һ����һ��ͼƬ
    output = []
    for i in range(bs):
        predictions = boxes[i]  # Ԥ��λ��xyxy  shape==(12700,85)
        score = np.max(predictions[:, 5:], axis=-1)
        # score = predictions[:,4]  # �����������Ŷ�,shape==12700
        mask = score > conf_thres  # �������Ŷ���ֵmask==[False,False,True......],shape==12700,True���ᱻ������False�н��ᱻɾ��
        detections = predictions[mask]  # ��һ��ɸѡ  shape==(115,85)
        class_conf = np.expand_dims(np.max(detections[:,5:],axis=-1),axis=-1)  # ��ȡÿ��Ԥ���Ԥ���������Ŷ�
        class_pred = np.expand_dims(np.argmax(detections[:,5:],axis=-1),axis=-1)  # ��ȡÿ��Ԥ��������±�
        # ����ѵ���(num_boxes,λ����Ϣ4+�����������1+������Ŷ�1+������1)
        detections = np.concatenate([detections[:,:4],class_conf,class_pred],axis=-1)  # shape=(numbox,7)

        unique_class = np.unique(detections[:,-1])  # ȡ���������������
        if len(unique_class)==0:
            continue
        best_box = []
        for c in unique_class:
            # ȡ�����Ϊc��Ԥ����
            cls_mask = detections[:,-1] == c
            detection = detections[cls_mask] # shape=(82,7)

            # �������������ʴӸ���������
            scores = detection[:,4]
            arg_sort = np.argsort(scores)[::-1]  # ���ص�������
            detection = detection[arg_sort]

            while len(detection) != 0:
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                # ���㵱ǰ���Ŷ����Ŀ������Ԥ����iou
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious < nms_thres]  # С��nms_thres����������ÿһ�����ټ���һ��
        output.append(best_box)

    boxes_loc = []
    conf_loc = []
    class_loc = []
    if len(output):
        for i in range(len(output)):
            pred = output[i]
            for i, det in enumerate(pred):
                if len(det):
                    # �������������ԭʼͼ����
                    det[0] = (det[0] - pad[0]) / ratio
                    det[2] = (det[2] - pad[0]) / ratio
                    det[1] = (det[1] - pad[1]) / ratio
                    det[3] = (det[3] - pad[1]) / ratio
                    boxes_loc.append([det[0],det[1],det[2],det[3]])
                    conf_loc.append(det[4])
                    class_loc.append(det[5])
    return boxes_loc,conf_loc,class_loc

def plot_one_box(img,boxes,conf,clas_id,line_thickness=3,names=None):
    # ��λ�ÿ�
    # tl = ����߿�  Ҫô����line_thicknessҪô����ԭͼim������Ϣ����Ӧ����һ��
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]),int(boxes[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # �������Ϣ��
    label = f'{names[int(clas_id)]} {conf:.2f}'
    tf = max(tl - 1, 1)  # label������߿� font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':

    command = ['ffmpeg',
               '-y',  # ��������ļ�
               '-f', 'rawvideo',  # �����ʽ
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',  # �������ظ�ʽ
               '-s', '1280x720',  # �����ͼ��ߴ�
               '-r', '30',  # �����֡��
               '-i', '-',  # �ӹܵ�����
               '-c:v', 'libx264',  # �����Ƶ����
               '-pix_fmt', 'yuv420p',  # ������ظ�ʽ
               '-preset', 'ultrafast',  # Ԥ��
               '-f', 'hls',  # �����ʽ
               '/usr/local/nginx/m3u8File/123.m3u8']  # ���RTMP��
               #'rtmp://47.122.30.74:1935/live2/123'
    p = sp.Popen(command, stdin=sp.PIPE)
    #os.environ["OMP_NUM_THREADS"] = "4"
    print("wait for openvino_model")
    names = ['fire','smoke']
    conf_thres = 0.5
    nms_thres = 0.4

    #img_path = r'C:\Users\25360\Desktop\people_test.webp.jpg'
    #input="E:/1_could_api/smoke_yolov5/data/images/input.mp4"
    STREAM_URL2 = 'rtmp://47.122.30.74:1935/live/123' #����
    input=STREAM_URL2
    fault = 0
    model_xml=r"/root/1_fire/yolov5-fire-detection/yolov5/2_ov/fire.xml"
    model_bin=r"/root/1_fire/yolov5-fire-detection/yolov5/2_ov/fire.bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    #˫�����Զ�����GPU��CPU��Դ��
    exec_net = ie.load_network(network=net, num_requests=2, device_name="HETERO:CPU")
    input_layer = next(iter(net.input_info))

    is_async_mode = False
    cap = cv2.VideoCapture(input)
    #fps = cap.get(cv2.CAP_PROP_FPS)  # ֡��
    #����ͷһ�ζ����֡����
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #�����ֵ��Ч��Ĭ����֡��Ϊ1��ʵ����ָ�޷���ȡ�����ܻ�ȡ�����ֵΪʵ������֡��
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    wait_key_code = 1
    #�첽ģʽ�¶����˵�ǰ֡����һ֡��֡ID��
    cur_request_id = 0
    next_request_id = 1


    is_async_mode = False
    wait_key_code = 0

    #������
    while True:
        print("getting cap")
        cap = cv2.VideoCapture(STREAM_URL2)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        if fps != 0:
            print("video is ok")
            # �첽ģʽ��ָ������һ��ͼƬ��AI������������ȴ�AI������������ֱ�Ӳɼ��ڶ���ͼƬ�������ͼ��Ԥ����Ȼ���ټ���һ��ͼƬ��������Ƿ���ϣ�����ϣ�������������
            # �����ĺô��ǣ�����ִ���˵�һ��ͼƬ��AI�����������͵ڶ���ͼƬ��ͼ��ɼ���Ԥ�������񣬼����˵ȴ�ʱ�䣬�����Ӳ���������ʣ�Ҳ�������������

                #����ͬ��ģʽ������֡�ṩ����ǰ����
            ret, frame = cap.read()
            t1 = time.time()
            while True:


                t1 = time.time()

                if not ret:
                    break
                request_id = cur_request_id
                img, ratio, (dw,dh) = letterbox(frame)

                # np.ascontiguousarray()��һ���ڴ治�����洢������ת��Ϊ�ڴ������洢�����飬ʹ�������ٶȸ���
                blob = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1/255.0, (img.shape[0], img.shape[1]), swapRB=True, crop=False)
                infer_request_handle=exec_net.start_async(request_id=request_id,inputs={input_layer: blob})
                #0�����������ѳ��֣�
                if infer_request_handle.wait(-1) == 0:
                    #����ֱ��ʹ���°�yolo5��3��1�����
                    res = infer_request_handle.output_blobs["output0"]
                    outs = res.buffer
                    boxes_loc,conf_loc,class_loc = non_max_suppression(outs, conf_thres=conf_thres, nms_thres=nms_thres,ratio=ratio, pad=(dw,dh))
                    # ���ӻ�
                    for i in range(len(boxes_loc)):
                        boxes = boxes_loc[i]
                        conf = conf_loc[i]
                        clas_id = class_loc[i]
                        plot_one_box(frame, boxes, conf, clas_id, line_thickness=3, names=names)
                #print(time.time() - t1)
                #cv2.imshow("result", frame)

                p.stdin.write(frame.tobytes())

                t2 = time.time()
                print(t2-t1)
                #key = (int((1 - (t2 - t1)) / int(fps)))*1000
               #key = (20-(t2-t1)*100)
                key=1
                #print(key)
                #if key>=0:
                cv2.waitKey(int(key))  # �ӳ�

                ret, frame = cap.read()
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                  #  print("break")
                  #  cap.release()
                   # cv2.destroyAllWindows()
                   # break

                if is_async_mode:
                    #�첽ģʽ�£���һ֡����ǰ֡����Ż�����
                    cur_request_id, next_request_id = next_request_id, cur_request_id
                    #frame = next_frame

                # ESC��������ʱ�˳�����
                if key == 27:
                   break
        else:
            print("can't get video,try again")
            # time.sleep(0.5)
            fault = fault + 1
            if fault >= 5:
                print("fault")
                break
            else:
                continue
    #whileѭ������ʱ�ͷ���Դ��
        #cv2.destroyAllWindows()
