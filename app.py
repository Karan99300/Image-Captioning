from PIL import Image
import requests
import gradio as gr 
import torch
from loader import get_loader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader,dataset=get_loader(root_folder='FlickrDataset/Images',annotation_file='FlickrDataset/Captions/captions.txt',transform=transform,num_workers=2)
filepath="ImageCaptioningusingLSTM.pth"
from model import CNNtoRNN
model=CNNtoRNN(embed_size=256,hidden_size=256,vocab_size=len(dataset.vocab),num_layers=1)
model.load_state_dict(torch.load(filepath))
model.eval()

def launch(input):
    im=Image.open(requests.get(input,stream=True).raw)
    image=transform(im.convert('RGB')).unsqueeze(0)
    
    return model.caption_image(image,dataset.vocab)

iface=gr.Interface(launch,inputs="text",outputs="text")
iface.launch()
    