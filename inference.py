import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
import pandas as pd
from loader import get_loader

def inference():
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_index=100
     
    train_loader,dataset=get_loader(root_folder='FlickrDataset/Images',annotation_file='FlickrDataset/Captions/captions.txt',transform=transform,num_workers=2)
    df=pd.read_csv("FlickrDataset/Captions/captions.txt")
    imagepath="FlickrDataset/Images/"
    images=os.listdir(imagepath)
    im=Image.open(os.path.join(imagepath,images[image_index]))
    im.show()

    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    filepath="ImageCaptioningusingLSTM.pth"
    model=CNNtoRNN(embed_size=256,hidden_size=256,vocab_size=len(dataset.vocab),num_layers=1).to(device)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    
    image=transform(im.convert("RGB")).unsqueeze(0)
    
    output=model.caption_image(image.to(device),dataset.vocab)
    predicted_caption = " ".join(output[1:-1])
    
    print("Reference Caption:", reference_caption)
    print("Predicted Caption:", predicted_caption)
    
if __name__=="__main__":
    inference()
