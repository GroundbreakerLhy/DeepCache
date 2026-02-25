import os
import urllib.request

target_dir = os.path.expanduser("~/onnx_zoo/")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

models = {
    "alexnet": "alexnet/model/bvlcalexnet-7.onnx",
    "squeezenet1.1": "squeezenet/model/squeezenet1.1-7.onnx",
    "mobilenetv2": "mobilenet/model/mobilenetv2-7.onnx",
    "googlenet": "inception_and_googlenet/googlenet/model/googlenet-7.onnx",
    "inception_v1": "inception_and_googlenet/inception_v1/model/inception-v1-7.onnx",
    "inception_v2": "inception_and_googlenet/inception_v2/model/inception-v2-7.onnx",
    "densenet121": "densenet-121/model/densenet-7.onnx",
    "resnet34": "resnet/model/resnet34-v1-7.onnx",
    "resnet50": "resnet/model/resnet50-v1-7.onnx",
    "resnet101": "resnet/model/resnet101-v1-7.onnx",
    "resnet152": "resnet/model/resnet152-v1-7.onnx",
    "vgg19": "vgg/model/vgg19-7.onnx",
    "shufflenet_v1": "shufflenet/model/shufflenet-9.onnx",
    "shufflenet_v2": "shufflenet/model/shufflenet-v2-10.onnx",
    "zfnet512": "zfnet-512/model/zfnet512-7.onnx",
    "caffenet": "caffenet/model/caffenet-7.onnx",
    "rcnn-ilsvrc13": "rcnn_ilsvrc13/model/rcnn-ilsvrc13-7.onnx",
    # Adding more models to reach ~30
    "mobilenetv1": "mobilenet/model/mobilenetv1-7.onnx",
    "efficientnet-lite4": "efficientnet-lite4/model/efficientnet-lite4-11.onnx",
    "inception_v3": "inception_and_googlenet/inception_v3/model/inception-v3-7.onnx",
    "resnet50-v2": "resnet/model/resnet50-v2-7.onnx",
}

# Victim models (not for training)
victim_models = {
    "resnet18": "resnet/model/resnet18-v1-7.onnx",
    "vgg16": "vgg/model/vgg16-7.onnx",
}

base_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/"

def download_set(model_dict, is_victim=False):
    for name, path in model_dict.items():
        url = base_url + path
        target_path = os.path.join(target_dir, f"{name}.onnx")
        prefix = "victim model " if is_victim else ""
        print(f"Downloading {prefix}{name} from {url}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"Successfully downloaded {prefix}{name}")
        except Exception as e:
            print(f"Failed to download {prefix}{name}: {e}")

# NLP/Transformer models from ONNX Model Zoo
nlp_models = {
    "bertsquad": "bert-squad/model/bertsquad-10.onnx",
    "gpt2": "gpt-2/model/gpt2-10.onnx",
    "roberta": "roberta/model/roberta-base-11.onnx",
    "t5-encoder": "t5/model/t5-encoder-12.onnx",
}

nlp_base_url = "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/"


def download_nlp_models():
    """Download NLP models from ONNX Model Zoo"""
    for name, path in nlp_models.items():
        url = nlp_base_url + path
        target_path = os.path.join(target_dir, f"{name}.onnx")
        print(f"Downloading {name} from {url}...")

        if os.path.exists(target_path):
            print(f"[Skip] {name} already exists")
            continue

        urllib.request.urlretrieve(url, target_path)
        print(f"Successfully downloaded {name}")


if __name__ == "__main__":
    # Comment out original CNN model downloads
    # download_set(models)
    # download_set(victim_models, is_victim=True)

    # Download NLP models from ONNX Model Zoo
    download_nlp_models()

    print("\nDisk space check:")
    os.system("df -h " + target_dir)
