import os
import json
from tqdm import tqdm
from efficient_ir import EfficientIR


NOTEXISTS = 'NOTEXISTS'


class Utils:

    def __init__(self, config):
        self.metainfo_path = config['metainfo_path']
        self.exists_index_path = config['exists_index_path']
        self.ir_engine = EfficientIR(
            config['img_size'],
            config['index_capacity'],
            config['index_path'],
            config['model_path'],
        )
        self.check_env()


    def check_env(self):
        if not os.path.exists(self.exists_index_path):
            parent_path = os.path.join(self.exists_index_path, os.pardir)
            os.makedirs(os.path.abspath(parent_path), exist_ok=True)
            with open(self.exists_index_path, 'w') as wp:
                wp.write("[]")


    def get_exists_index(self):
        return json.loads(open(self.exists_index_path, 'rb').read())


    def get_file_list(self, target_dir):
        accepted_exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp']
        file_path_list = []
        for root, dirs, files in os.walk(target_dir):
            for name in files:
                if name.lower().endswith(tuple(accepted_exts)):
                    file_path_list.append(os.path.join(root, name))
        return file_path_list


    def index_target_dir(self, target_dir):
        # 如果已有文件索引就加载
        exists_index = []
        if os.path.exists(self.exists_index_path):
            exists_index = json.loads(open(self.exists_index_path, 'rb').read())
        # 如果已有元信息索引就加载
        metainfo = []
        if os.path.exists(self.metainfo_path):
            metainfo = json.loads(open(self.metainfo_path, 'rb').read())
        # 枚举当前指定目录的所有文件全路径
        this_index = self.get_file_list(target_dir)
        # 需要特征索引的文件
        need_index = []
        # 更新文件索引
        for i in this_index:
            if not i in exists_index:
                exists_index.append(i)
        # 更新元信息索引
        for i in range(len(exists_index)):
            if NOTEXISTS in exists_index[i]:
                continue
            # 采集元信息
            file_size = os.path.getsize(exists_index[i])
            file_mtime = os.path.getmtime(exists_index[i])
            # 新增元信息
            if i >= len(metainfo):
                metainfo.append([file_size, file_mtime])
                need_index.append(i)
                continue
            # 检查元信息更新
            if metainfo[i][0] != file_size or metainfo[i][1] != file_mtime:
                metainfo[i] = [file_size, file_mtime]
                need_index.append(i)
        # 写入索引文件
        with open(self.exists_index_path, 'wb') as wp:
            wp.write(json.dumps(exists_index,ensure_ascii=False).encode('UTF-8'))
        with open(self.metainfo_path, 'wb') as wp:
            wp.write(json.dumps(metainfo,ensure_ascii=False).encode('UTF-8'))
        return [(i,exists_index[i]) for i in need_index]


    def update_ir_index(self, need_index):
        for idx, fpath in tqdm(need_index, ascii=True, desc='更新索引记录'):
            fv = self.ir_engine.get_fv(fpath)
            if fv is None:
                continue
            self.ir_engine.add_fv(fv, idx)
        self.ir_engine.save_index()


    def remove_nonexists(self):
        exists_index = []
        if os.path.exists(self.exists_index_path):
            exists_index = json.loads(open(self.exists_index_path, 'rb').read())
        for idx in tqdm(range(len(exists_index)), ascii=True, desc='删除不存在文件'):
            if not os.path.exists(exists_index[idx]):
                exists_index[idx] = NOTEXISTS
                self.ir_engine.hnsw_index.mark_deleted(idx)
        with open(self.exists_index_path, 'wb') as wp:
            wp.write(json.dumps(exists_index,ensure_ascii=False).encode('UTF-8'))


    def checkout(self, image_path, exists_index, match_n=5):
        fv = self.ir_engine.get_fv(image_path)
        sim, ids = self.ir_engine.match(fv, match_n)
        return [(sim[i], exists_index[ids[i]]) for i in range(len(ids))]


    def get_duplicate(self, exists_index, threshold, same_folder):
        matched = set()
        for idx in tqdm(range(len(exists_index)), ascii=True, desc='检索重复图像中'):
            match_n = 5
            try:
                fv = self.ir_engine.hnsw_index.get_items([idx])[0]
            except RuntimeError:
                continue
            sim, ids = self.ir_engine.match(fv, match_n)
            while sim[-1] > threshold:
                match_n = round(match_n*1.5)
                sim, ids = self.ir_engine.match(fv, match_n)
            for i in range(len(ids)):
                if ids[i] == idx:
                    continue
                if sim[i] < threshold:
                    continue
                if ids[i] in matched:
                    continue
                if not idx in matched:
                    matched.add(idx)
                path_a = exists_index[idx]
                path_b = exists_index[ids[i]]
                if same_folder:
                    if os.path.dirname(path_a) != os.path.dirname(path_b):
                        continue
                yield (path_a, path_b, sim[i])
