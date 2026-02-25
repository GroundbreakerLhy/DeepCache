import numpy as np
from PIL import Image

image_length = 256

block = 3

width = 64 * block
height = image_length * block

# trace_file = "./cache_dataset/cache_dataset_llc_tvm/alexnet/0018.tvmgen_default_fused_layout_transform_3_compute_-2930-2a9f-.npy"

# data = np.load(trace_file)
# a, b, c, d = data.shape
# print("data shape:", data.shape)
# data = data.reshape(a * b * c, d)

# skip = 0
# 截取要显示的片段
# cur_pic = data[skip : skip + image_length]

# img = Image.new(mode="L", size=(width, height))


trace_file = "experiment_llc/cache_dataset/cache_dataset_tvm/bvlcalexnet-3/0039.tvmgen_default_fused_nn_contrib_dense_pack_add_compute_-0x406280-0x406371.log"
skip = 3800

with open(trace_file, "r") as f:
    counter = 0
    while counter < skip:
        line = f.readline()
        if not line:
            break
        counter += 1

    cur_pic = []
    while len(cur_pic) < height:
        for i in range(image_length):
            line = f.readline()
            if not line:
                break
            if not (line.startswith("0") or line.startswith("1")):
                break
            vec = line.strip().split()
            vec = [int(c) for c in vec]
            cur_pic.append(vec)

img = Image.new(mode="L", size=(width, height))
pixels = img.load()

for i in range(image_length):
    for j in range(64):
        # 1 代表 Miss (深色)，0 代表 Hit (亮色)
        target = 120 if cur_pic[i][j] else 230
        for b1 in range(block):
            for b2 in range(block):
                pixels[j * block + b1, i * block + b2] = target

img.save("trace_visualize.jpg")
