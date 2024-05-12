import agorartc
from openvino.inference_engine import IECore
import random
import ctypes
import time
from flask import Flask, render_template, Response
import numpy as np
import cv2
from PIL import Image
from agoraToken import agoraTokenSetting
import pdb
from mqtts.drone_state_get import droneMQTTClient

from multiprocessing import Process
from multiprocessing import Queue

stream = 0
frame_index = 0
app = Flask(__name__)
exec_net = None
input_layer = None
kinds = None

def mqtt_thread(img_queue):
    print("mqtt_thread started")
    #global mqtt_flag,save_img
    while True:
        time.sleep(0.1)
        save_img=img_queue.get()
        #print(save_img)
        if save_img is not None:
            time.sleep(1)
            print("save")
            #mqttclient.connect()
            #mqttclient.fireImg = save_img
            #mqttclient.loop_forever()
            #mqttclient.lastCall = current_time

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), scaleup=False, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    # (for better test mAP) scale_up = False 对于大于new_shape（r<1）的原图进行缩放,小于new_shape（r>1）的不变
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
    # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
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

#非极大值抑制函数
def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.4, ratio=1, pad=(20,20)):
    # 取出batch_size
    bs = np.shape(boxes)[0]
    # xywh___ to____ xyxy
    shape_boxes = np.zeros_like(boxes[:,:,:4])
    shape_boxes[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
    shape_boxes[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
    shape_boxes[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
    shape_boxes[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2
    boxes[:, :, :4] = shape_boxes
    boxes[:, :, 5:] *= boxes[:, :, 4:5]

    # output存放每一张图片的预测结果，推理阶段一般是一张图片
    output = []
    for i in range(bs):
        predictions = boxes[i]  # 预测位置xyxy  shape==(12700,85)
        score = np.max(predictions[:, 5:], axis=-1)
        # score = predictions[:,4]  # 存在物体置信度,shape==12700
        mask = score > conf_thres  # 物体置信度阈值mask==[False,False,True......],shape==12700,True将会被保留，False列将会被删除
        detections = predictions[mask]  # 第一次筛选  shape==(115,85)
        class_conf = np.expand_dims(np.max(detections[:,5:],axis=-1),axis=-1)  # 获取每个预测框预测的类别置信度
        class_pred = np.expand_dims(np.argmax(detections[:,5:],axis=-1),axis=-1)  # 获取每个预测框的类别下标
        # 结果堆叠，(num_boxes,位置信息4+包含物体概率1+类别置信度1+类别序号1)
        detections = np.concatenate([detections[:,:4],class_conf,class_pred],axis=-1)  # shape=(numbox,7)

        unique_class = np.unique(detections[:,-1])  # 取出包含的所有类别
        if len(unique_class)==0:
            continue
        best_box = []
        for c in unique_class:
            # 取出类别为c的预测结果
            cls_mask = detections[:,-1] == c
            detection = detections[cls_mask] # shape=(82,7)

            # 包含物体类别概率从高至低排列
            scores = detection[:,4]
            arg_sort = np.argsort(scores)[::-1]  # 返回的是索引
            detection = detection[arg_sort]

            while len(detection) != 0:
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                # 计算当前置信度最大的框和其它预测框的iou
                ious = iou(best_box[-1],detection[1:])
                detection = detection[1:][ious < nms_thres]  # 小于nms_thres将被保留，每一轮至少减少一个
        output.append(best_box)

    boxes_loc = []
    conf_loc = []
    class_loc = []
    if len(output):
        for i in range(len(output)):
            pred = output[i]
            for i, det in enumerate(pred):
                if len(det):
                    # 将框坐标调整回原始图像中
                    det[0] = (det[0] - pad[0]) / ratio
                    det[2] = (det[2] - pad[0]) / ratio
                    det[1] = (det[1] - pad[1]) / ratio
                    det[3] = (det[3] - pad[1]) / ratio
                    boxes_loc.append([det[0],det[1],det[2],det[3]])
                    conf_loc.append(det[4])
                    class_loc.append(det[5])
    return boxes_loc,conf_loc,class_loc

def plot_one_box(img,boxes,conf,clas_id,line_thickness=3,names=None):

    # 画位置框
    # tl = 框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
    #global mqtt_flag, save_img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = (0,255,0)
    c1 = (int(boxes[0]), int(boxes[1]))
    c2 = (int(boxes[2]), int(boxes[3]))
    img = img.copy()

    cv2.rectangle(img, c1, c2, color, thickness=3, lineType=cv2.LINE_AA)

    # 画类别信息框
    label = f'{names[int(clas_id)]} {conf:.2f}'
    tf = max(tl - 1, 1)  # label字体的线宽 font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    current_time = time.time()

    return img,current_time


def init_img(width, height, ybuffer):
    rgba_array = (ctypes.c_ubyte * (width * height * 4)).from_address(ybuffer)
    im = Image.frombuffer('RGBA', (width, height), rgba_array, 'raw', 'RGBA', 0, 1)
    frame = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    img, ratio, (dw, dh) = letterbox(frame) #重绘大小
    blob = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1 / 255.0, (img.shape[0], img.shape[1]),
                                 swapRB=True, crop=False)
    return blob,ratio,dw,dh

def show_img(infer_request_handle,conf_thres,nms_thres,kinds,ratio,dw,dh):
    global stream
    if infer_request_handle.wait(-1) == 0:
        # 这里直接使用新版yolo5的3合1输出：
        res = infer_request_handle.output_blobs["output0"]
        outs = res.buffer
        boxes_loc, conf_loc, class_loc = non_max_suppression(outs, conf_thres=conf_thres,
                                                             nms_thres=nms_thres, ratio=ratio, pad=(dw, dh))
        # 可视化
        for i in range(len(boxes_loc)):
            boxes = boxes_loc[i]
            conf = conf_loc[i]
            clas_id = class_loc[i]
            frame = plot_one_box(frame, boxes, conf, clas_id, line_thickness=3, names=kinds)
        stream = frame.copy()


def detection_mul(detect_queue):
    mqttclient = droneMQTTClient()
    print("wait for openvino_model")
    kinds = ['fire', 'smoke']
    conf_thres = 0.5
    nms_thres = 0.4
    fault = 0
    model_xml = r"./model/fire.xml"
    model_bin = r"./model/fire.bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    # 双请求，自动分配GPU及CPU资源：
    exec_net = ie.load_network(network=net, num_requests=2, device_name="HETERO:CPU")
    input_layer = next(iter(net.input_info))
    f=[0,0]
    s=[0,0]
    t=[0,0]
    flag=0
    while True:
        input = detect_queue.get()
        if input is not None:
            #print("get input")
            width=input[0]
            height=input[1]
            rgba_array=input[2]
            #print(input)
            #print(f,s,t)
            if flag==0:
                print("first out")

                im = Image.frombuffer('RGBA', (width, height), rgba_array, 'raw', 'RGBA', 0, 1)
                f[0] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                img, ratio, (dw, dh) = letterbox(f[0])  # 重绘大小
                f[1] = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1 / 255.0, (img.shape[0], img.shape[1]),
                                             swapRB=True, crop=False)
                #(f)
                print("first out2")
                flag=1
                detect_queue.put([None,0])  # 输出

            elif flag==1:
                s[0]=f[0]
                s[1] = exec_net.start_async(request_id=0, inputs={input_layer: f[1]})

                im = Image.frombuffer('RGBA', (width, height), rgba_array, 'raw', 'RGBA', 0, 1)
                f[0] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                img, ratio, (dw, dh) = letterbox(f[0])  # 重绘大小
                f[1] = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1 / 255.0, (img.shape[0], img.shape[1]),
                                          swapRB=True, crop=False)
                print("second out")
                flag=2
                detect_queue.put([None,0])  # 输出

            elif f is not None and s is not None :
                if s[1].wait(-1) == 0:
                    t[0]=s[0]
                    t[1] = s[1].output_blobs["output0"]

                    s[0] = f[0]
                    s[1] = exec_net.start_async(request_id=0, inputs={input_layer: f[1]})  # 再次启动推理，更新s

                    outs = t[1].buffer
                    boxes_loc, conf_loc, class_loc = non_max_suppression(outs, conf_thres=conf_thres,
                                                                         nms_thres=nms_thres, ratio=ratio, pad=(dw, dh))
                    # 可视化
                    current_time=0
                    for i in range(len(boxes_loc)):
                        boxes = boxes_loc[i]
                        conf = conf_loc[i]
                        clas_id = class_loc[i]
                        t[0],current_time = plot_one_box(t[0], boxes, conf, clas_id, line_thickness=3, names=kinds)
                    detect_queue.put((t[0],current_time)) #输出 也可以直接画在上一帧的图像上

                    im = Image.frombuffer('RGBA', (width, height), rgba_array, 'raw', 'RGBA', 0, 1)
                    f[0] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                    img, ratio, (dw, dh) = letterbox(f[0])  # 重绘大小
                    f[1] = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1 / 255.0, (img.shape[0], img.shape[1]),
                                              swapRB=True, crop=False)


def streampb():
    global stream
    while True:
        if stream is not None:
            _, buffer = cv2.imencode('.jpg', stream)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # time.sleep(1 / 60)  # 控制发送速度
@app.route('/')
def index():
    # 返回包含视频流的HTML页面
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 返回视频流响应
    return Response(streampb(), mimetype='multipart/x-mixed-replace; boundary=frame')\


class MyVideoFrameObserver(agorartc.VideoFrameObserver):
    def onCaptureVideoFrame(self, width, height, ybuffer, ubuffer, vbuffer):
        t1=time.time()
        rgba_array = (ctypes.c_ubyte * (width * height * 4)).from_address(ybuffer)
        rgba_array_np = np.ctypeslib.as_array(rgba_array)
        detect_queue.put((width,height,rgba_array_np))
        global stream
        stream=detect_queue.get()
        print(time.time()-t1)

    def onRenderVideoFrame(self, uid, width, height, ybuffer, ubuffer, vbuffer):
        t1=time.time()

        rgba_array = (ctypes.c_ubyte * (width * height * 4)).from_address(ybuffer)
        rgba_array_np = np.ctypeslib.as_array(rgba_array)
        detect_queue.put((width,height,rgba_array_np))

        global stream
        detect_output=detect_queue.get()
        current_time=detect_output[1]
        if (current_time - mqttclient.lastCall) >= 5:
            mqttclient.lastCall = current_time
            stream = detect_output[0]
            img_queue.put(stream)
        else:
            stream = detect_output[0]
            #print(stream)
        print(time.time()-t1)


def joinChannel():

    APPID = "fae36796eb8946d2897b34df9530d068"
    rtc.initialize(APPID, None, agorartc.AREA_CODE_GLOB & 0xFFFFFFFF)
    rtc.enableVideo()
    rtc.enableLocalVideo(False)  # 禁用本地视频
    rtc.enableLocalAudio(False)
    token = agoraTokenSetting(7788)
    rtc.joinChannel(token, "test", "", 7788)
    rtc.startPreview()
    agorartc.registerVideoFrameObserver(rtc, videoFrameObserver)
# Agora SDK初始化和事件处理逻辑...
# 注意：这里需要根据你的实际需求来初始化Agora SDK，加入频道等

if __name__ == '__main__':
    # 初始化Agora SDK和事件监听等
    img_queue =Queue()
    mqttclient = droneMQTTClient()
    #detect_init()

    mqtt=Process(target=mqtt_thread,args=(img_queue,))
    mqtt.start()

    detect_queue=Queue()

    fire_detect=Process(target=detection_mul,args=(detect_queue,))
    fire_detect.start()


    rtc = agorartc.createRtcEngineBridge()
    event_handler = agorartc.RtcEngineEventHandlerBase()  # 或者是你自定义的事件处理类
    rtc.initEventHandler(event_handler)
    videoFrameObserver = MyVideoFrameObserver()
    joinChannel()

    # 这里添加加入Agora频道的代码
    app.run(debug=False,host='0.0.0.0',port=6060)
    mqtt.join()
    fire_detect.join()

