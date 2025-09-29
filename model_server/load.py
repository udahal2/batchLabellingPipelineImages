import torch
import numpy as np
from PIL import Image
import torchvision
import json
import matplotlib.pyplot as plt
import cv2

with open('class_mapping.json') as data:
    mappings = json.load(data)

class_mapping = {item['model_idx']: item['class_name'] for item in mappings}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load('model.pt').to(device)

image_path = 'data/sample_images/image8.jpeg'
image = Image.open(image_path)
# Transform your image if the config.yaml shows
# you used any image transforms for validation data
image = np.array(image)
# Convert to torch tensor
x = torch.from_numpy(image).to(device)
with torch.no_grad():
    # Convert to channels first, convert to float datatype
    x = x.permute(2, 0, 1).float()
    y = model(x)

    # Some optional postprocessing, you can change the 0.5 iou
    # overlap as needed
    to_keep = torchvision.ops.nms(y['pred_boxes'], y['scores'], 0.5)
    y['pred_boxes'] = y['pred_boxes'][to_keep]
    y['pred_classes'] = y['pred_classes'][to_keep]

    # Draw you box predictions:
    for bbox, label in zip(y['pred_boxes'], y['pred_classes']):
        bbox = list(map(int, bbox))
        x1, y1, x2, y2 = bbox
        class_idx = label.item()
        class_name = class_mapping[class_idx]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(
            image,
            class_name,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (255, 0, 0)
        )
# Display predicted boxes and classes on your image
plt.imshow(image)
plt.show()