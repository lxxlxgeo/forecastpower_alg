from torch import nn
import torch
from torch.nn.modules.activation import MultiheadAttention
import torch.nn.functional as F







class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''
    def __init__(self,kernelsize,stride,pandding):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=2,out_channels=1,kernel_size=kernelsize,stride=stride, padding=pandding, bias=False)
        #self.bn1=nn.BatchNorm3d(1)
        self.sigmoid = nn.Tanh()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # min_out,_=torch.min(x,dim=1,keepdim=True)

        x = torch.cat([max_out,avg_out], dim=1)
        x = self.conv1(x)
        # x = self.bn1(x)
        x=self.sigmoid(x)
        return x


class ChannelAttention(nn.Module):
    '''
    通道注意力模块
    '''
    def __init__(self,in_planes,ratio=16) -> None:
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool3d(1)
        self.max_pool=nn.AdaptiveMaxPool3d(1)
        
        
        self.conv1=nn.Conv3d(in_channels=in_planes,out_channels=in_planes//ratio,kernel_size=(1,1,1),bias=False)
        #self.relu1=nn.ReLU()
        self.conv2=nn.Conv3d(in_channels=in_planes//ratio,out_channels=in_planes,kernel_size=(1,1,1),bias=False)
        #self.bn1=nn.BatchNorm3d(in_planes)
        self.sigmoid=nn.Tanh()
    
    def forward(self,x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        avg_out = self.conv2(self.conv1(avg_out))
        max_out = self.conv2(self.conv1(max_out))
        
        out = avg_out + max_out
        
        #out = self.bn1(out)  # Batch Normalization
        out = self.sigmoid(out)
        return out
    
    
class Residual_7_7(nn.Module):  #@save
    '''
    残差连接模块
    '''
    
    def __init__(self, input_channels, num_channels, use_1x1conv=False,use_sp=False, strides=(1,1,1)):
        super().__init__()
        
        #self.ch_at=ChannelAttention(input_channels,input_channels)
        #self.sp_at=SpatialAttention()
        self.conv1 = nn.Conv3d(input_channels,num_channels, kernel_size=(3,3,3), padding=(1,1,1), stride=strides)
        
        self.conv1_bn=nn.BatchNorm3d(num_channels)
        #self.conv1_relu=nn.ReLU()
        
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=(7,7,7), padding=(3,3,3))
        self.conv2_relu=nn.ReLU()
        self.conv2_bn=nn.BatchNorm3d(num_channels)
        
        if use_sp:
            self.ch_at0=ChannelAttention(num_channels,num_channels)
            self.sp_at0=SpatialAttention(kernelsize=3,stride=1,pandding=1)
        else:
            self.ch_at0=None
            self.sp_at0=None
        # self.bn1=nn.BatchNorm3d(num_channels)
        # self.bn2 = nn.BatchNorm3d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, num_channels, kernel_size=(1,1,1),padding=(0,0,0), stride=strides)
            self.resnet_relu=nn.ReLU()
            
        else:
            self.conv3 = None
        #self.bn1 = nn.BatchNorm3d(num_channels)
        

    def forward(self, X):
        
        identity=X
        out=self.conv1(X)
        out=self.conv1_bn(out)
        #out=self.conv1_relu(out)
        
        out=self.conv2(out)
        out=self.conv2_bn(out)
        out=self.conv2_relu(out)
        
        if self.ch_at0:
            out=self.ch_at0(out)*out
            out=self.sp_at0(out)*out
            
        
        if self.conv3:
            identity = self.conv3(X)
            out=out+identity
            out=self.resnet_relu(out)
            return out
        else:
            return out

class Residual(nn.Module):
    '''
    残差连接模块
    '''
    
    def __init__(self, input_channels, num_channels, use_1x1conv=False, use_sp=False, \
        use_ch=False,use_pool=False, strides=(1,1,1),sa_k=3,sa_s=1,sa_p=1,activate_f='relu'):
        super().__init__()
        
        self.conv1 = nn.Conv3d(input_channels, num_channels, kernel_size=(3,3,3), padding=(1,1,1), stride=strides)
        self.conv1_bn = nn.BatchNorm3d(num_channels)
        #self.conv1_relu = nn.ReLU()
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2_bn = nn.BatchNorm3d(num_channels)
        self.conv2_relu = nn.ReLU()

        if use_sp:
            self.sp_at0 = SpatialAttention(kernelsize=sa_k,stride=sa_s,pandding=sa_p)
        else:      
            self.sp_at0 = None
            
        if use_ch:
            self.ch_at0=ChannelAttention(num_channels,num_channels)
        else:
            self.ch_at0=None 
            

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, num_channels, kernel_size=(3,3,3), padding=(1,1,1), stride=strides)
            if use_pool:
                self.use_pool = True
                self.max_pool_conv = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
                self.resnet_relu =nn.ReLU()
            else: 
                self.use_pool = False
                self.resnet_relu =nn.ReLU()
        else:
            if use_pool:
                self.max_pool_conv = nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
                self.conv3 = None
            else:
                self.conv3 = None

    def forward(self, X):
        identity = X
        out = self.conv1(X)
        out = self.conv1_bn(out)
        #out = self.conv1_relu(out)

        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.conv2_relu(out)

        if self.sp_at0:
            out= self.sp_at0(out) * out
        if self.ch_at0:
            out=self.ch_at0(out)*out

        if self.conv3:
            identity = self.conv3(X)
            out = out + identity

            if self.use_pool:
                out = self.max_pool_conv(out)
                out = self.resnet_relu(out)
                return out
            else:
                out = self.resnet_relu(out)
                return out
        else:
            # if not self.use_pool:
            #     out = self.max_pool_conv(out)
            #     return out
            # else:
            return out
        
class WeatherCNN3D(nn.Module):
    def __init__(self,mattention=0.3,mlp_drop=0.3,fc=256,num_heads=16,at_fun='relu'):
        super(WeatherCNN3D, self).__init__()
        '''
        外罗三期
        '''
        
        self.mattention=mattention #注意力drop
        self.mlp_drop=mlp_drop #全连接drop 
        self.fc=fc #全连接通道数
        self.num_heads=num_heads #注意力头
        if at_fun=='relu':
            self.linear_at_fun=nn.ReLU()
        elif at_fun=='sigmoid':
            self.linear_at_fun=nn.Sigmoid()
        elif at_fun=='tanh':
            self.linear_at_fun=nn.Tanh()
        
        # 20——>10*10--->5*5--->3*3
        #init 卷积层
        self.resnet1=Residual_7_7(18,32,False,False,strides=(1,2,2))  #10*10 
        
        # 3*3*3 卷积层
        self.resnet3_2=Residual(32,64,True,use_sp=False,use_ch=False,use_pool=False,strides=(1,2,2))#,sa_k=7,sa_s=1,sa_p=3) 
        
        # 3*3*3 卷积层
        # self.resnet3_3=Residual(64,64,True,use_sp=True,use_ch=True,use_pool=False,strides=(1,1,1),sa_k=5,sa_s=1,sa_p=2)
        
        # 3*3*3 卷积层
        self.resnet3_5= Residual(64,128,True,use_sp=False,use_ch=False,use_pool=False,strides=(1,2,2)) # 5*5 
        
        # 3*3*3 卷积层
        # self.resnet3_8= Residual(128,256,True,False,False,False,strides=(1,1,1))
        
        # 3*3*3 卷积层
        self.resnet5_3=Residual(128,512,True,use_sp=False,use_ch=False,use_pool=False,strides=(1,2,2))
        
        self.fc_layer=nn.Linear(512*2*2,self.fc)
        
        self.attention1 = MultiheadAttention(embed_dim=self.fc,num_heads=self.num_heads,dropout=0.0,batch_first=True)
        self.layer_norm=nn.LayerNorm(self.fc)
        
        self.attention1_pool=nn.Linear(self.fc,1)
        self.attention1_pool_relu=nn.ReLU()
        
        
        # self.relu_fc=nn.Tanh()
        self.mlp_dropout=nn.Dropout(self.mlp_drop)
        self.fc3=nn.Linear(self.fc, 10)
        
        
    def forward(self, x):
        
        x=self.resnet1(x)
        x=self.resnet3_2(x)
        # x=self.resnet3_3(x)
        x=self.resnet3_5(x)
        # x=self.resnet3_8(x)
        x=self.resnet5_3(x)
        
        x = x.view(x.size(0), x.size(2), -1)
        
        x=self.fc_layer(x)

        attn_output, at_weights = self.attention1(x,x,x)
        attn_output=x+attn_output
        attn_output=self.layer_norm(attn_output)        
        
        attn_wights=self.attention1_pool(attn_output).squeeze(-1)
        attn_wights=self.attention1_pool_relu(attn_wights)
        
        #attn_wights=self.at_pool_drop(attn_wights)
        attn_wights=F.softmax(attn_wights,dim=1)
        x=torch.sum(attn_output*attn_wights.unsqueeze(-1),dim=1)        
        
        #x,_=self.attention_mlp(x,x,x)
        #输出最终结果
        #x=self.relu_fc(x)
        x=self.mlp_dropout(x)
        x=self.fc3(x)
        x = x.squeeze(dim=1)
        return x
