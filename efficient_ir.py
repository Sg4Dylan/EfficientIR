import os
import numpy as np
from PIL import Image
import hnswlib
import onnxruntime


class EfficientIR:

    def __init__(self, img_size, index_capacity, index_path, model_path):
        self.img_size = img_size
        self.index_capacity = index_capacity
        self.index_path = index_path
        self.model_path = model_path
        self.init_index()
        self.load_index()
        self.init_model()
        Image.MAX_IMAGE_PIXELS = None


    def img_preprocess(self, image_path):
        try:
            img = Image.open(image_path).resize((self.img_size, self.img_size),Image.BICUBIC)
            img = img.convert('RGBA').convert('RGB')
        except OSError:
            print(f'\nFile broken: {image_path}')
            return None
        input_data = np.array(img).transpose(2, 0, 1)
        # 预处理
        img_data = input_data.astype('float32')
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        # add batch channel
        norm_img_data = norm_img_data.reshape(1, 3, self.img_size, self.img_size).astype('float32')
        return norm_img_data


    def init_index(self):
        self.hnsw_index = hnswlib.Index(space='l2', dim=1000)
        return self.hnsw_index


    def load_index(self):
        if os.path.exists(self.index_path):
            self.hnsw_index.load_index(self.index_path, max_elements=self.index_capacity)
        else:
            self.hnsw_index.init_index(max_elements=self.index_capacity, ef_construction=200, M=48)


    def save_index(self):
        self.hnsw_index.save_index(self.index_path)


    def init_model(self):
        self.session_opti = onnxruntime.SessionOptions()
        self.session_opti.enable_mem_pattern = False
        self.session = onnxruntime.InferenceSession(self.model_path, self.session_opti)
        # self.session.set_providers(['DmlExecutionProvider'])
        self.model_input = self.session.get_inputs()[0].name
        return self.session, self.model_input


    def get_fv(self, image_path):
        norm_img_data = self.img_preprocess(image_path)
        if norm_img_data is None:
            return None
        return self.session.run([], {self.model_input: norm_img_data})[0][0]


    def add_fv(self, fv, idx):
        self.hnsw_index.add_items(fv, idx)


    def match(self, fv, nc=5):
        query = self.hnsw_index.knn_query(fv, k=nc)
        similarity = (1-np.tanh(query[1][0]/3000))*100
        return similarity, query[0][0]
