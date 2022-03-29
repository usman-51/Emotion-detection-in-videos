# from feat.detector import Detector
import wget

# def detection(image_path = '', video_path=''):
#     try:
#         face_model = "retinaface"
#         landmark_model = "mobilenet"
#         au_model = "rf"
#         # emotion_model = "resmasknet"
#         emotion_model = "fer"
#         detector = Detector( face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)
#         # Detect FEX from image
#         if image_path:
#             out = detector.detect_image(image_path)
#             predictions_mean = out.emotions().mean(axis=0)
#             return predictions_mean

#         # Detect FEX from video
#         if video_path:
#             out = detector.detect_video(video_path, skip_frames=24)
#             predictions_mean = out.emotions().mean(axis=0)
#             return predictions_mean

#         return "File path is not given"
#     except Exception as e:
#       print('Exceptoin ',e)
#       return False
    

def detection(video_path='', detector=None):
    try:
        # Detect FEX from video
        if video_path:
            out = detector.detect_video(video_path, skip_frames=24)
            predictions_mean = out.emotions().mean(axis=0)
            return predictions_mean

        return "File path is not given"
    except Exception as e:
        print('Exceptoin ', e)
        return False

def download_file(url):
    vid_name = wget.download(url)
    return vid_name
