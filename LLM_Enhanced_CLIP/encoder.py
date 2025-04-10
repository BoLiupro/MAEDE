from huggingface_hub import hf_hub_download
import torch, open_clip
from PIL import Image
from IPython.display import display
import os
import pandas as pd

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

model_name = 'RN50'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

path_to_your_checkpoints = ''

ckpt = torch.load(f"{path_to_your_checkpoints}/ImageEncoder-{model_name}.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)
model = model.cuda().eval()
image_folder = ""
output_csv = ""

#Read all images in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png'))]

data = []

#Disable gradient calculation to accelerate inference
with torch.no_grad(), torch.cuda.amp.autocast():
    for file_name in image_files:
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        img_feat = image_features.cpu().numpy().flatten().tolist()  # (1,1024) → [1024]
        data.append([file_name, img_feat])
        print(f"finish:{file_name}\n")

df = pd.DataFrame(data, columns=["file_name", "image_features"])
df.to_csv(output_csv, index=False)