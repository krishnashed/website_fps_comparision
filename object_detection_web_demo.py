#!/usr/bin/env python3
"""
 Copyright (C) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from flask import Flask, render_template, Response
import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

app = Flask(__name__)


def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    return frame


def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))


def gen_frames(device):
    # args = build_argparser().parse_args()
    if 'ssd' != 'yolov4' and None:
        log.warning('The "--anchors" option works only for "-at==yolov4". Option will be omitted')
    if 'ssd' != 'yolov4' and None:
        log.warning('The "--masks" option works only for "-at==yolov4". Option will be omitted')
    if 'ssd' not in ['nanodet', 'nanodet-plus'] and None:
        log.warning('The "--num_classes" option works only for "-at==nanodet" and "-at==nanodet-plus". Option will be omitted')

    cap = open_images_capture('/home/intel/open_model_zoo/demos/object_detection_demo/python/vehicles.mp4', False)


    plugin_config = get_user_config(device, '1', None)
    model_adapter = OpenvinoAdapter(create_core(), '/home/intel/intel/vehicle-detection-0202/FP16-INT8/vehicle-detection-0202.xml', device=device, plugin_config=plugin_config,
                                    max_num_requests=0, model_parameters = {'input_layouts': None})

    configuration = {
        'resize_type': None,
        'mean_values': None,
        'scale_values': None,
        'reverse_input_channels': False,
        'path_to_labels': '/home/intel/open_model_zoo/data/dataset_classes/voc_20cl_bkgr.txt',
        'confidence_threshold': 0.5,
        'input_size': (600, 600), # The CTPN specific
        'num_classes': None, # The NanoDet and NanoDetPlus specific
    }
    model = DetectionModel.create_model('ssd', model_adapter, configuration)
    model.log_layers_info()

    detector_pipeline = AsyncPipeline(model)

    next_frame_id = 0
    next_frame_id_to_show = 0

    palette = ColorPalette(len(model.labels) if model.labels else 100)
    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    # video_writer = cv2.VideoWriter()

    while True:
        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(objects) and False:
                print_raw_results(objects, model.labels, next_frame_id_to_show)

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_detections(frame, objects, palette, model.labels, output_transform)
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)

            # if video_writer.isOpened() and (1000 <= 0 or next_frame_id_to_show <= 1000-1):
            #     video_writer.write(frame)
            next_frame_id_to_show += 1

            if not None:
                # cv2.imshow('Detection Results', frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

                key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            continue

        if detector_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                output_transform = OutputTransform(frame.shape[:2], None)
                if None:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])
                presenter = monitors.Presenter('', 55,
                                               (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                # if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                #                                          cap.fps(), output_resolution):
                    # raise RuntimeError("Can't open video writer")
            # Submit for inference
            detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        else:
            # Wait for empty request
            detector_pipeline.await_any()

    detector_pipeline.await_all()
    if detector_pipeline.callback_exceptions:
        raise detector_pipeline.callback_exceptions[0]
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = detector_pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(objects) and False:
            print_raw_results(objects, model.labels, next_frame_id_to_show)

        presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        frame = draw_detections(frame, objects, palette, model.labels, output_transform)
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)

        # if video_writer.isOpened() and (1000 <= 0 or next_frame_id_to_show <= 1000-1):
        #     video_writer.write(frame)

        if not None:
            # cv2.imshow('Detection Results', frame)
            key = cv2.waitKey(1)

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            presenter.handleKey(key)

    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          detector_pipeline.preprocess_metrics.get_latency(),
                          detector_pipeline.inference_metrics.get_latency(),
                          detector_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/cpu-benchmark')
def cpu_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames('CPU'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gpu-benchmark')
def gpu_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames('GPU.1'), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)
