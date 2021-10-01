import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import pushup
import pullup
import situp
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


class VideoCamera(object):
    def __init__(self, type=None):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.6)
        self.type = None
        self.seq_list = []
        self.state = False
        self.counter = 0
        #Case untuk type
        if type == 'pushup':
            ##Insert model pushup
            self.type = 'pushup'
            self.pose_embedding = pushup.FullBodyPoseEmbedder()
        elif type == 'situp':
            ##Insert model situp
            self.type = 'situp'
            self.model = self.open_situp_model()
            
            self.pose_embedding = situp.FullBodyPoseEmbedder()
        elif type == 'pullup':
            ##Insert model pullup
            self.type = 'pullup'
            self.pose_embedding = pullup.FullBodyPoseEmbedder()
        else:
            self.model = None

    def __del__(self):
        self.video.release()

    def seq_check(self, seq):
        ''' if the sequence is ['up', 'down', 'up'],
        it is considered as a valid sequence. Hence, the
        counter is added. Other than that the counter not
        added. the list that passed in this function
        is never empty'''

        if seq[0] == 'down' and len(seq) == 1:
            return seq.clear(), False
        elif seq[0] == 'up' and len(seq) == 1:
            return seq, False

        if len(seq) == 2:
            if seq[0] == seq[1]:
                seq.pop(0)
            return seq, False

        if len(seq) == 3:
            if seq[1] == seq[2]:
                seq.pop(1)
                return seq, False
            else:
                return seq.clear(), True

    def open_pushup_model(self):
        # open model pushup
        model_path = './models/pushups/model_6/pushups.csv'

        # import csv file and store in pandas
        df = pd.read_csv(model_path)
        df = df.drop(columns='Unnamed: 0')

        # Label Encoder
        le = LabelEncoder()
        y = le.fit_transform(df['push_up_motion'])

        # drop 'picture_name' and 'push_up_motion'
        df_copy = df.copy()
        df_copy = df_copy.drop(columns=['picture_name', 'push_up_motion'])
        X = df_copy.to_numpy()
        return X, y

    def open_situp_model(self):
        # situp model path
        model_path = './models/situps/embedded_situp.csv'
        
        df = pd.read_csv(model_path)
        df['target'] = df['situp_position'].apply(lambda x: 0 if x == 'situp_down' else 1)
        
        x = df[situp.feature_columns].values
        y = df['target'].values
        
        x /= 180.0
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(x,y)
        return knn_classifier

    def open_pullup_model(self):
        # open model pushup
        model_path = './models/pullups/model2_pullup.csv'

        # import csv file and store in pandas
        df = pd.read_csv(model_path)
        df = df.drop(columns='Unnamed: 0')

        # Label Encoder
        le = LabelEncoder()
        y = le.fit_transform(df['pull_up_motion'])

        # drop 'picture_name' and 'push_up_motion'
        df_copy = df.copy()
        df_copy = df_copy.drop(columns=['picture_name', 'pull_up_motion'])
        X = df_copy.to_numpy()
        return X, y

    def get_frame(self):
        success, input_frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

        result = self.pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks
        if self.type == None:        
            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
                pose_landmarks = np.array([ [lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark], dtype=np.float32)
                # output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        elif self.type == 'pushup':
            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
                pose_landmarks = np.array([ [lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark], dtype=np.float32)
                X, y = self.open_pushup_model()
                embedding = self.pose_embedding(pose_landmarks)
                my_knn = pushup.KNNClassifier(X, y, embedding, K=5)
                dict_result, distances_result = my_knn()
                if dict_result["up"] > dict_result["down"] and dict_result['conf_level'] > 50:
                    cv2.putText(output_frame, 'up ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.seq_list.append('up')
                    _, self.state = self.seq_check(self.seq_list)
                    if self.state:
                        self.counter += 1
                elif dict_result["down"] > dict_result["up"] and dict_result['conf_level'] > 50:
                    cv2.putText(output_frame, 'down ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.seq_list.append('down')
                    _, self.state = self.seq_check(self.seq_list)
                else:
                    cv2.putText(output_frame, 'not detected ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output_frame, 'Count: ' + str(self.counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output_frame, 'Push-up ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif self.type == 'situp':
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
                pose_landmarks = np.array([ [lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark], dtype=np.float32)
                embedding = self.pose_embedding(pose_landmarks)
                embedding = embedding.reshape(1,-1)
                embedding = embedding/180.0
                prediction = self.model.predict_proba(embedding)

                if prediction[0][1] >= 0.8:
                    cv2.putText(output_frame, 'up ' + str(100*prediction[0][1]) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.seq_list.append('up')
                    _, self.state = self.seq_check(self.seq_list)

                elif prediction[0][0] >= 0.8 :
                    cv2.putText(output_frame, 'down ' + str(prediction[0][0]*100) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.seq_list.append('down')
                    _, self.state = self.seq_check(self.seq_list)
                    if self.state:
                        self.counter += 1
                        self.seq_list = []
                else:
                    cv2.putText(output_frame, 'not detected ' + str(100*prediction[0][0]) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output_frame, 'Count: ' + str(self.counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output_frame, 'Sit-up ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            pass
        elif self.type == 'pullup':
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
                pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark],
                                        dtype=np.float32)

                # process embedding and make classifications
                X, y = self.open_pullup_model()
                embedding = self.pose_embedding(pose_landmarks)
                my_knn = pullup.KNNClassifier(X, y, embedding, K=5)
                dict_result, distances_result = my_knn()

                if dict_result["up"] > dict_result["down"] and dict_result['conf_level'] > 50:
                    cv2.putText(output_frame, 'up ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.seq_list.append('up')
                    _, self.state = self.seq_check(self.seq_list)

                elif dict_result["down"] > dict_result["up"] and dict_result['conf_level'] > 50:
                    cv2.putText(output_frame, 'down ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.seq_list.append('down')
                    _, self.state = self.seq_check(self.seq_list)
                    if self.state:
                        self.counter += 1
                        self.seq_list = []
                else:
                    cv2.putText(output_frame, 'not detected ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output_frame, 'Count: ' + str(self.counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(output_frame, 'Pull-up ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', output_frame)
        print(type(jpeg.tobytes()))
        return jpeg.tobytes()

