from os import path
from utils import video_emotion_detection
from flask_restful import Resource, Api, reqparse
from feat.detector import Detector
from flask import Flask, request
from bs4 import BeautifulSoup
from flask_cors import CORS
from flask import jsonify

import pandas as pd
import numpy as np
import random
import time,os

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = True

api = Api(app)


class video_emotions(Resource):
    def post(self):
        try:
            video_path = request.form.get('video_path')
            vid_name = ''
            if not 'home' in video_path:
                vid_name = video_emotion_detection.download_file(video_path)
                video_path = vid_name

            # loading emotion detector
            # face_model = "retinaface"
            # face_model = "img2pose"
            face_model = "FaceBoxes"
            # face_model = "img2pose"
            landmark_model = "mobilenet"
            au_model = "rf"
            # emotion_model = "resmasknet"
            emotion_model = "fer"
            detector = Detector(face_model=face_model, landmark_model=landmark_model,
                                au_model=au_model, emotion_model=emotion_model)

            ###################################
            emotion_pred = video_emotion_detection.detection(video_path, detector)
            try:
                emotion_indexes = emotion_pred.index.to_list()
                emotion_scores = list(emotion_pred)
                dict_emotions = {}
                for ind, val in enumerate(emotion_indexes):
                    dict_emotions[val] = emotion_scores[ind]

                dict_emotions = sorted(dict_emotions.items(),
                                    key=lambda x: x[1], reverse=True)
                dict_emotions = dict(dict_emotions)

            except:
                dict_emotions = {'Error': "File path is not Given"}
            #####################################

            # # spliting and recognising face emotions
            # for i in questions.keys():
            #     if i.startswith('Ans'):
            #         clip = VideoFileClip(video_path).subclip(
            #             questions[i][0], questions[i][1])
            #         number = str(random.randint(0, 9999))
            #         clip_name = vid_name.split(
            #             '_')[-1].split('-')[-1].split('.')[0] + '_' + number + '_' + i + '.mp4'
            #         clip.write_videofile(clip_name)
            #         emotion_pred = video_emotion_detection.detection(clip_name, detector)
            #         try:
            #             emotion_indexes = emotion_pred.index.to_list()
            #             emotion_scores = list(emotion_pred)
            #             dict_emotions = {}
            #             for ind, val in enumerate(emotion_indexes):
            #                 dict_emotions[val] = emotion_scores[ind]

            #             dict_emotions = sorted(dict_emotions.items(),
            #                                 key=lambda x: x[1], reverse=True)
            #             dict_emotions = dict(dict_emotions)

            #         except:
            #             dict_emotions = {'Error': "File path is not Given"}

            #         all_ans_emotions[i] = dict_emotions
            #         os.remove(clip_name)

            # if vid_name:
            #     os.remove(vid_name)

            pred={'emotions': dict_emotions}
            return pred
        except Exception as e:
            # if vid_name:
            #     os.remove(vid_name)
            return {'emotions': e}, 400



api.add_resource(video_emotions, '/video_emotions', methods=['POST'])

if __name__ == "__main__":
    app.run(debug=True,port=8099)