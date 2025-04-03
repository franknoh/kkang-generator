import torch
import cv2
import numpy as np
import gradio as gr
from src.segement import Segmenter
from src.CFA import CFA
from torchvision import transforms
from PIL import Image, ImageDraw

num_landmark = 24
img_width = 128
s = 640
crop_pad = 0.1
mask_scale = 0.65

def rmbg_fn(img):
    input_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    input_img = (input_img / 255).astype(np.float32)

    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor)
    with torch.no_grad():
        pred = segmenter(tmpImg).squeeze(0).cpu().numpy()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]

    if np.sum(pred) == 0:
        mask = img[:, :, 3]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask, img.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = np.concatenate((pred * img + 1 - pred, pred * 255), axis=2).astype(np.uint8)

    # img의 alpha가 0인 부분 crop
    x_min, x_max = np.where(pred[:, :, 0] == 1)[1].min(), np.where(pred[:, :, 0] == 1)[1].max()
    y_min, y_max = np.where(pred[:, :, 0] == 1)[0].min(), np.where(pred[:, :, 0] == 1)[0].max()
    img = img[y_min:y_max, x_min:x_max]

    pred = pred * 255
    pred = np.clip(pred, 0, 255)
    pred = pred.astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    return pred, img

def landmark_fn(img):
    faces = face_detector.detectMultiScale(img)

    if len(faces) == 0:
        print("No face detected")
        return img.copy(), img.copy()
    
    alpha = img[:, :, 3]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
    draw = ImageDraw.Draw(img)
    img_crop = img.copy()
    img_crop = img_crop.convert("RGBA")
    img_crop.putalpha(Image.fromarray(alpha))

    x_, y_, w_, h_ = faces[0]
    x = max(x_ - w_ / 8, 0)
    rx = min(x_ + w_ * 9 / 8, img.width)
    y = max(y_ - h_ / 4, 0)
    by = y_ + h_
    w = rx - x
    h = by - y

    img_crop = img_crop.crop((int(x-crop_pad * w), int(y-crop_pad * h), int(x+w+crop_pad * w), int(y+h+crop_pad * h)))
    img_crop = img_crop.resize((img_width, img_width), Image.BICUBIC)
    img_crop = np.array(img_crop)

    draw.rectangle((x, y, x + w, y + h), outline=(0, 0, 255), width=3)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    img_tmp = img.crop((x, y, x+w, y+h))
    img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)
    img_tmp = transform(img_tmp)
    img_tmp = img_tmp.unsqueeze(0)

    heatmap = landmark_detector(img_tmp)
    heatmap = heatmap.cpu().detach().numpy()[0]

    for i in range(num_landmark):
        heatmap_tmp = cv2.resize(heatmap[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmap_tmp), heatmap_tmp.shape)
        landmark_y = landmark[0] * h / img_width
        landmark_x = landmark[1] * w / img_width

        draw.ellipse((x + landmark_x - 2, y + landmark_y - 2, x + landmark_x + 2, y + landmark_y + 2), fill=(255, 0, 0))
    
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    return img, img_crop

def morph_fn(img):
    h, w = img.shape[:2]
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    size = max(w, h)
    img = cv2.copyMakeBorder(img, 0, size-h, (size-w)//2, (size-w)//2, cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))

    h, w = img.shape[:2]
    
    img = cv2.copyMakeBorder(img, int(h/4), 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))
    height, width = img.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = 2 * (x - width/2) / width
    ny = 2 * (y - height/8) / height
    r = np.sqrt(nx*nx + ny*ny)
    effect = np.exp(-10 * r*r)
    horiz_factor = np.exp(-10 * nx*nx)
    push = effect * horiz_factor
    map_x = x
    map_y = y - push * height * 2
    
    pinch_factor = 0.2 * effect
    map_x = map_x + (width/2 - map_x) * pinch_factor
    
    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        result[:,:,c] = cv2.remap(img[:,:,c], map_x, map_y, 
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    
    result = result[height-h:, :, :]

    return result

def add_bg_fn(img):
    template = cv2.imread('assets/template.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGBA)
    
    h, w = template.shape[:2]
    img = cv2.resize(img, (int(w * mask_scale), int(h * mask_scale)), interpolation=cv2.INTER_CUBIC)
    
    x_offset = (w - img.shape[1]) // 2 - 10
    y_offset = h - img.shape[0] -22
    
    output = template.copy()
    for c in range(3):
        output[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1], c] = \
            (img[:, :, c] * (img[:, :, 3] / 255.0) + 
             template[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1], c] * 
             (1 - img[:, :, 3] / 255.0)).astype(np.uint8)

    return output

def main(img):
    seg_mask, seg_img = rmbg_fn(img)
    landmark_img, landmark_crop = landmark_fn(seg_img)
    morphed_img = morph_fn(landmark_crop)
    out_img = add_bg_fn(morphed_img)

    return seg_mask, seg_img, landmark_img, landmark_crop, morphed_img, out_img



if __name__ == "__main__":
    segmenter = Segmenter()
    face_detector = cv2.CascadeClassifier('models/lbpcascade.xml')
    landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name='models/landmark.pth')

    app = gr.Blocks()
    with app:
        with gr.Column():
            input_img = gr.Image(label="input image", image_mode="RGBA", width=384)
            examples_data = ["examples/test3.png", "examples/test4.png", "examples/test5.png"]
            examples = gr.Examples(examples=examples_data, inputs=[input_img])
            run_btn = gr.Button(variant="primary")

        with gr.Row():
            output_mask = gr.Image(label="mask", format="png")
            output_segment = gr.Image(label="segmented image", image_mode="RGBA", format="png")
            output_landmark = gr.Image(label="landmark image", image_mode="RGBA", format="png")

        with gr.Row():
            output_crop = gr.Image(label="landmark crop", image_mode="RGBA", format="png")
            output_morphed = gr.Image(label="morphed image", image_mode="RGBA", format="png")
            output_final = gr.Image(label="final image", image_mode="RGB", format="png")
        
        run_btn.click(main, [input_img], [output_mask, output_segment, output_landmark, output_crop, output_morphed, output_final])
    app.launch()