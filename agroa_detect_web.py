import ctypes
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
import sys
import MainWindow
import agorartc
from PyQt5 import QtOpenGL
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import cv2
from PIL import Image
from openvino.inference_engine import IECore
import random
import time
import datetime
remoteWinId = -1
from flask import Flask, render_template, Response
import threading

stream = 0
frame_index = 0
app = Flask(__name__)
exec_net = None
input_layer = None
kinds = None

# 线程锁
stream_lock = threading.Lock()

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), scaleup=False, stride=32):
    """
    将图片缩放调整到指定大小,1920x1080的图片最终会缩放到640x384的大小，和YOLOv4的letterbox不一样
    Resize and pad image while meeting stride-multiple constraints
    :param img: 原图 hwc
    :param new_shape: 缩放后的最长边大小
    :param color: pad的颜色
    :param auto: True：进行矩形填充  False：直接进行resize
    :param scale_up: True：仍进行上采样 False：不进行上采样
    :return: img: letterbox后的图片 HWC
             ratio: wh ratios
             (dw, dh): w和h的pad
    """
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
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]),int(boxes[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # 画类别信息框
    label = f'{names[int(clas_id)]} {conf:.2f}'
    tf = max(tl - 1, 1)  # label字体的线宽 font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def detect_init():
    global exec_net
    global input_layer
    global kinds
    print("wait for openvino_model")
    kinds = ['fire','smoke']
    #conf_thres = 0.5
    #nms_thres = 0.4

    fault = 0
    model_xml = r"E:/1_could_api/fire_detect/2_ov/fire.xml"
    model_bin = r"E:/1_could_api/fire_detect/2_ov/fire.bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    # 双请求，自动分配GPU及CPU资源：
    exec_net = ie.load_network(network=net, num_requests=2, device_name="HETERO:CPU")
    input_layer = next(iter(net.input_info))

def detection(frame,exec_net,input_layer,conf_thres,nms_thres,kinds):
    global stream
    t1=time.time()
    img, ratio, (dw, dh) = letterbox(frame) #重绘大小

    # np.ascontiguousarray()将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    blob = cv2.dnn.blobFromImage(np.ascontiguousarray(img), 1 / 255.0, (img.shape[0], img.shape[1]),
                                 swapRB=True, crop=False)
    infer_request_handle = exec_net.start_async(request_id=0, inputs={input_layer: blob})
    # 0代表推理结果已出现：
    if infer_request_handle.wait(-1) == 0:

        res = infer_request_handle.output_blobs["output0"]
        outs = res.buffer
        boxes_loc, conf_loc, class_loc = non_max_suppression(outs, conf_thres=conf_thres,
                                                             nms_thres=nms_thres, ratio=ratio, pad=(dw, dh))
        # 可视化
        for i in range(len(boxes_loc)):
            boxes = boxes_loc[i]
            conf = conf_loc[i]
            clas_id = class_loc[i]
            print("result",clas_id)
            plot_one_box(frame, boxes, conf, clas_id, line_thickness=3, names=kinds)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #cv2.imshow("result", frame)

    #print("检测耗时：",(time.time() - t1))
    key = 1
    cv2.waitKey(int(key))  # 延迟
    #with stream_lock:
    stream = frame.copy()

class MyRtcEngineEventHandler(agorartc.RtcEngineEventHandlerBase):

    def __init__(self, rtc):
        super().__init__()
        self.rtc = rtc

    def onWarning(self, warn, msg):
        print("warn: ")
        print(warn)

    def onError(self, err, msg):
        print("err: ")
        print(err)

    def onJoinChannelSuccess(self, channel, uid, elapsed):
        print("onJoinChannelSuccess")

    def onRejoinChannelSuccess(self, channel, uid, elapsed):
        print("onRejoinChannelSuccess")

    def onLeaveChannel(self, stats):
        print("onLeaveChannel")

    def onClientRoleChanged(self, oldRole, newRole):
        print("onClientRoleChanged")

    def onUserJoined(self, uid, elapsed):
        global remoteWinId
        if remoteWinId != -1:
            remoteVideoCanvas = agorartc.createVideoCanvas(remoteWinId)
            remoteVideoCanvas.uid = uid
            self.rtc.setupRemoteVideo(remoteVideoCanvas)

    def onUserOffline(self, uid, reason):
        print("onUserOffline")

    def onLastmileQuality(self, quality):
        print("onLastmileQuality")

    def onLastmileProbeResult(self, res):
        print("onLastmileProbeResult")

    def onConnectionInterrupted(self):
        print("onConnectionInterrupted")

    def onConnectionLost(self):
        print("onConnectionLost")

    def onConnectionBanned(self):
        print("onConnectionBanned")

    def onApiCallExecuted(self, err, api, res):
        print("onApiCallExecuted" + str(api))

    def onRequestToken(self):
        print("onRequestToken")

    def onTokenPrivilegeWillExpire(self, token):
        print("onTokenPrivilegeWillExpire")

    def onAudioQuality(self, uid, quality, delay, lost):
        print("onAudioQuality")

    def onRtcStats(self, stats):
        print("onRtcStats")

    def onNetworkQuality(self, uid, txQuality, rxQuality):
        print("onNetworkQuality")

    def onLocalVideoStats(self, stats):
        print("onLocalVideoStats")

    def onRemoteVideoStats(self, stats):
        print("onRemoteVideoStats")

    def onLocalAudioStats(self, stats):
        print("onLocalAudioStats")

    def onRemoteAudioStats(self, stats):
        print("onRemoteAudioStats")

    def onLocalAudioStateChanged(self, state, error):
        print("onLocalAudioStateChanged")

    def onRemoteAudioStateChanged(self, uid, state, reason, elapsed):
        print("onRemoteAudioStateChanged")

    def onAudioVolumeIndication(self, speakers, speakerNumber, totalVolume):
        print("onAudioVolumeIndication")

    def onActiveSpeaker(self, uid):
        print("onActiveSpeaker")

    def onVideoStopped(self):
        print("onVideoStopped")

    def onFirstLocalVideoFrame(self, width, height, elapsed):
        print("onFirstLocalVideoFrame")

    def onFirstRemoteVideoDecoded(self, uid, width, height, elapsed):
        print("onFirstRemoteVideoDecoded")

    def onFirstRemoteVideoFrame(self, uid, width, height, elapsed):
        print("onFirstRemoteVideoFrame")

    def onUserMuteAudio(self, uid, muted):
        print("onUserMuteAudio")

    def onUserMuteVideo(self, uid, muted):
        print("onUserMuteVideo")

    def onUserEnableVideo(self, uid, enabled):
        print("onUserEnableVideo")

    def onAudioDeviceStateChanged(self, deviceId, deviceType, deviceState):
        print("onAudioDeviceStateChanged")

    def onAudioDeviceVolumeChanged(self, deviceType, volume, muted):
        print("onAudioDeviceVolumeChanged")

    def onCameraReady(self):
        print("onCameraReady")

    def onCameraFocusAreaChanged(self, x, y, width, height):
        print("onCameraFocusAreaChanged")

    def onCameraExposureAreaChanged(self, x, y, width, height):
        print("onCameraExposureAreaChanged")

    def onAudioMixingFinished(self):
        print("onAudioMixingFinished")

    def onAudioMixingStateChanged(self, state, errorCode):
        print("onAudioMixingStateChanged")

    def onRemoteAudioMixingBegin(self):
        print("onRemoteAudioMixingBegin")

    def onRemoteAudioMixingEnd(self):
        print("onRemoteAudioMixingEnd")

    def onAudioEffectFinished(self, soundId):
        print("onAudioEffectFinished")

    def onFirstRemoteAudioDecoded(self, uid, elapsed):
        print("onFirstRemoteAudioDecoded")

    def onVideoDeviceStateChanged(self, deviceId, deviceType, deviceState):
        print("onVideoDeviceStateChanged")

    def onLocalVideoStateChanged(self, localVideoState, error):
        print("onLocalVideoStateChanged")

    def onVideoSizeChanged(self, uid, width, height, rotation):
        print("onVideoSizeChanged")

    def onRemoteVideoStateChanged(self, uid, state, reason, elapsed):
        print("onRemoteVideoStateChanged")

    def onStreamMessage(self, uid, streamId, data, length):
        print("onStreamMessage")

    def onStreamMessageError(self, uid, streamId, code, missed, cached):
        print("onStreamMessageError")

    def onMediaEngineLoadSuccess(self):
        print("onMediaEngineLoadSuccess")

    def onMediaEngineStartCallSuccess(self):
        print("onMediaEngineStartCallSuccess")

    def onChannelMediaRelayStateChanged(self, state, code):
        print("onChannelMediaRelayStateChanged")

    def onChannelMediaRelayEvent(self, code):
        print("onChannelMediaRelayEvent")

    def onFirstLocalAudioFrame(self, elapsed):
        print("onFirstLocalAudioFrame")

    def onFirstRemoteAudioFrame(self, uid, elapsed):
        print("onFirstRemoteAudioFrame")

    def onRtmpStreamingStateChanged(self, url, state, errCode):
        print("onRtmpStreamingStateChanged")

    def onStreamPublished(self, url, error):
        print("onStreamPublished")

    def onStreamUnpublished(self, url):
        print("onStreamUnpublished")

    def onTranscodingUpdated(self):
        print("onTranscodingUpdated")

    def onStreamInjectedStatus(self, url, uid, status):
        print("onStreamInjectedStatus")

    def onAudioRouteChanged(self, routing):
        print("onAudioRouteChanged")

    def onLocalPublishFallbackToAudioOnly(self, isFallbackOrRecover):
        print("onLocalPublishFallbackToAudioOnly")

    def onRemoteSubscribeFallbackToAudioOnly(self, uid, isFallbackOrRecover):
        print("onRemoteSubscribeFallbackToAudioOnly")

    def onRemoteAudioTransportStats(self, uid, delay, lost, rxKBitRate):
        print("onRemoteAudioTransportStats")

    def onRemoteVideoTransportStats(self, uid, delay, lost, rxKBitRate):
        print("onRemoteVideoTransportStats")

    def onMicrophoneEnabled(self, enabled):
        print("onMicrophoneEnabled")

    def onConnectionStateChanged(self, state, reason):
        print("onConnectionStateChanged")

    def onNetworkTypeChanged(self, type):
        print("onNetworkTypeChanged")

    def onLocalUserRegistered(self, uid, userAccount):
        print("onLocalUserRegistered")

    def onUserInfoUpdated(self, uid, info):
        print("onUserInfoUpdated")

class MyVideoFrameObserver(agorartc.VideoFrameObserver):
    def onCaptureVideoFrame(self, width, height, ybuffer, ubuffer, vbuffer):
        pass
    def onRenderVideoFrame(self,uid, width, height, ybuffer, ubuffer, vbuffer):
        rgba_array = (ctypes.c_ubyte * (width * height * 4)).from_address(ybuffer)
        im = Image.frombuffer('RGBA', (width, height), rgba_array, 'raw', 'RGBA', 0, 1)
        frame = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        #print("---------------detection and save-------------------")
        detection(frame, exec_net, input_layer, 0.5, 0.4, kinds)
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


class MyWindow():
    def __init__(self):
       
        self.rtc = agorartc.createRtcEngineBridge()
        self.eventHandler = MyRtcEngineEventHandler(self.rtc)
        self.rtc.initEventHandler(self.eventHandler)
        self.videoFrameObserver = MyVideoFrameObserver()
        self.token="007eJxTYGiZEROUeGlCzOFA3VnT5BrSN7n0t97yWnJ9y/bkDZZuy10VGFJSDYwsjExTkgwszE0MQbRhiqWhiaEZkGNsYpGUsfBGakMgI8NS7stMjAwQCOIzMxgaGTMwAAApjB4y"


    def joinChannel(self):
        APPID = "de02825db087415db01d91416415348b"
        self.rtc.initialize(APPID, None, agorartc.AREA_CODE_GLOB & 0xFFFFFFFF)
        self.rtc.enableVideo()
        #token = "007eJxTYGiuEimed/bGya2yhzdrp4Zfr5Z/8iDr8vVd+RYXfkxJ+NukwJCWmGpsZm5plppkYWlilmJkYWmeZGySkmZpamyQYmBmcWzZtdSGQEYG6fM9zIwMEAjiszCUpBaXMDAAAEWLIvI="
        #token = "007eJxTYGiZEROUeGlCzOFA3VnT5BrSN7n0t97yWnJ9y/bkDZZuy10VGFJSDYwsjExTkgwszE0MQbRhiqWhiaEZkGNsYpGUsfBGakMgI8NS7stMjAwQCOIzMxgaGTMwAAApjB4y"
        self.rtc.joinChannel(self.token, "123", "", 0)
        self.rtc.startPreview()
        agorartc.registerVideoFrameObserver(self.rtc, self.videoFrameObserver)



def streampb():
    global stream
    while True:
        if stream is not None:
            _, buffer = cv2.imencode('.jpg', stream)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1 / 60)  # 控制发送速度
@app.route('/')
def index():
    # 返回包含视频流的HTML页面
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 返回视频流响应
    return Response(streampb(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    detect_init()
    app2 = QApplication(sys.argv)
    window = MyWindow()
    window.joinChannel()
    app.run(debug=True, host='0.0.0.0', port=6060)
    sys.exit(app2.exec_())
