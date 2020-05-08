from utils import *
import time


# 索引测试
image_dir = 'YOUR_IMAGE_DIR'                                    # 图片仓库的文件夹

# 第一次使用或新加入图片，使用这一段代码构建索引
remove_nonexists()
exists_index = index_target_dir(image_dir)                      # 对目标文件夹包含图片做递归路径索引
update_ir_index(exists_index)                                   # 按路径索引顺序建立内容检索索引


# 检索测试
exists_index = get_exists_index()                               # 获取目标文件夹的路径索引记录
start_time = time.time()
for i in os.listdir('target_img'):                              # 放置等待被检索的图片
    results = checkout(f'target_img/{i}', exists_index, 2)      # 被检索图片路径，图片仓库的索引，返回结果的数量
    print(f'Input: {i}')
    for result in results:
        print(f'Similarity: {result[0]:.2f} % Matched: {result[1]}')
print(f'Match cost: {time.time()-start_time:.4f}s')
