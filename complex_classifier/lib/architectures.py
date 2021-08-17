#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:07:04 2021

@author: siegfriedkaidisch

Pytorch ANN architectures
"""
import torch
import torch.nn as nn


class FC1(torch.nn.Module):
    '''
    ANN Architecture with 1 hidden, fully connected layer
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_1,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc_out(x)
        return x
    

class FC2(torch.nn.Module):
    '''
    ANN Architecture with 2 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_2,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc_out(x)
        return x
    

class FC6(torch.nn.Module):
    '''
    ANN Architecture with 6 hidden, fully connected layers
    '''
    
    def __init__(self,
                 ### Dataset-specific parameters
                 in_features: int,
                 out_features: int,
 
                 ### Fully connected layers
                 hidden_dim_1: int, 
                 hidden_dim_2: int, 
                 hidden_dim_3: int, 
                 hidden_dim_4: int, 
                 hidden_dim_5: int, 
                 hidden_dim_6: int, 
                 
                 *args,
                 **kwargs
                 ):
        
        # init superclasss
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features,
                              out_features=hidden_dim_1)
        
        self.fc2 = nn.Linear(in_features=hidden_dim_1,
                              out_features=hidden_dim_2)
        
        self.fc3 = nn.Linear(in_features=hidden_dim_2,
                              out_features=hidden_dim_3)
        
        self.fc4 = nn.Linear(in_features=hidden_dim_3,
                             out_features=hidden_dim_4)
        
        self.fc5 = nn.Linear(in_features=hidden_dim_4,
                             out_features=hidden_dim_5)
        
        self.fc6 = nn.Linear(in_features=hidden_dim_5,
                             out_features=hidden_dim_6)
    
        self.fc_out = nn.Linear(in_features=hidden_dim_6,
                              out_features=out_features)
        
        
    def forward(self, x):        
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        
        x = self.fc4(x)
        x = torch.relu(x)
        
        x = self.fc5(x)
        x = torch.relu(x)
        
        x = self.fc6(x)
        x = torch.relu(x)

        x = self.fc_out(x)
        return x
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #