import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loader import get_loader
from model import CNNtoRNN
from tqdm import tqdm
from tqdm import trange

def train():
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
    train_loader,dataset=get_loader(root_folder='FlickrDataset/Images',annotation_file='FlickrDataset/Captions/captions.txt',transform=transform,num_workers=2)
    
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    embed_size=256
    hidden_size=256
    vocab_size=len(dataset.vocab)
    num_layers=1
    learning_rate=3e-4
    num_epochs=200
    
    model=CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    train_iterator=trange(0,num_epochs)
    for _ in train_iterator:
        pbar=tqdm(train_loader)
        for idx,(imgs,captions) in enumerate(pbar):
            model.train()
            imgs=imgs.to(device)
            captions=captions.to(device)
            
            outputs=model(imgs,captions[:-1])
            loss=criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))
            
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item())
    
    filepath="ImageCaptioningusingLSTM.pth"        
    torch.save(model.state_dict(),filepath)
    

            
if __name__=="__main__":
    train()
            
                        
        
    
    
    
    
    