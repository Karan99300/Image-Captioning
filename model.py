import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self,embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch= nn.BatchNorm1d(embed_size,momentum = 0.01)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self,images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderRNN, self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.dropout=nn.Dropout(0.5)
        
    def forward(self,features,captions):
        embeddings=self.dropout(self.embed(captions))
        embeddings=torch.cat((features.unsqueeze(0),embeddings),dim=0)
        hiddens,_=self.lstm(embeddings)
        outputs=self.linear(hiddens)
        
        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN,self).__init__()
        self.encoderCNN=EncoderCNN(embed_size)
        self.decoderRNN=DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)
        
    def forward(self,images,captions):
        features=self.encoderCNN(images)
        outputs=self.decoderRNN(features,captions)
        return outputs
    
    def caption_image(self,image,vocabulary,max_length=50):
        result_caption=[]
        with torch.no_grad():
            X=self.encoderCNN(image).unsqueeze(0)
            states=None
            
            for _ in range(max_length):
                hiddens,states=self.decoderRNN.lstm(X,states)
                output=self.decoderRNN.linear(hiddens.squeeze(0))
                predicted=output.argmax(1)
                result_caption.append(predicted.item())
                
                X=self.decoderRNN.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()]=="<EOS>":
                    break
                
        return [vocabulary.itos[idx] for idx in result_caption]

        
        
    