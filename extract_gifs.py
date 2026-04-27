import os
from PIL import Image

def extract_frames(gif_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Image.open(gif_path)
    i = 0
    while True:
        img.save(os.path.join(output_dir, f"frame_{i:d}.png"), "PNG")
        i += 1
        try:
            img.seek(i)
        except EOFError:
            break
    print(f"Extracted {i} frames from {gif_path} to {output_dir}")

base_dir = "plots"
gifs = [
    "LaplaceModel_anim_1.gif",
    "SVDSurrogate_anim_1.gif",
    "LaplaceLatentModel_anim_1.gif",
    "LaplaceSVDModel_anim_1.gif",
    "ae_study_anim_0.gif",
    "ch4_demo.gif"
]

for g in gifs:
    gif_path = os.path.join(base_dir, g)
    if os.path.exists(gif_path):
        out_dir = os.path.join(base_dir, "frames_" + g.replace(".gif", ""))
        # Only run if dir doesn't exist or doesn't have 150 frames
        if not os.path.exists(out_dir) or len(os.listdir(out_dir)) < 150:
            extract_frames(gif_path, out_dir)
        else:
            print(f"Skipping {g}, already extracted")
