import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel

# ----------------------------
# Configuration
# ----------------------------
CKPT_DIR = "./ofa-large-caption"   # local model path
IMAGE_DIR = "/home/aayush_shah/ppStruct/paddlex_hps_PP-StructureV3_sdk/combined_outputJuspayPaddlebackend/images2/" # change this to your folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Preprocessing
# ----------------------------
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ----------------------------
# Load model & tokenizer
# ----------------------------
print("Loading model...")
tokenizer = OFATokenizer.from_pretrained(CKPT_DIR)
model = OFAModel.from_pretrained(CKPT_DIR, use_cache=False).to(DEVICE)
# Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("OFA-Sys/ofa-large-caption", torch_dtype="auto")
# Load model directly


# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")
# model.eval()

# ----------------------------
# Captioning loop
# ----------------------------
OUTPUT_FILE = "captions_output.txt"
prompt = " what does the image describe?"

with open(OUTPUT_FILE, 'w') as f:
    f.write("OFA-Large Caption Results\n")
    f.write("=" * 80 + "\n\n")
    
    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        path = os.path.join(IMAGE_DIR, filename)
        image = Image.open(path)
        patch_img = patch_resize_transform(image).unsqueeze(0).to(DEVICE)

        # Tokenize text prompt
        inputs = tokenizer([prompt], return_tensors="pt").input_ids.to(DEVICE)

        # Generate caption
        with torch.no_grad():
            gen = model.generate(
                inputs,
                patch_images=patch_img,
                num_beams=5,
                max_length=20,
                no_repeat_ngram_size=3,
            )
            caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

        result = f"{filename} → {caption}"
        print(result)
        f.write(result + "\n")

print(f"\nResults saved to {OUTPUT_FILE}")
