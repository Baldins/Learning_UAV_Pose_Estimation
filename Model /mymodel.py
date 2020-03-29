class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# class UnFlatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1, 1, 1)


class FlattenIMU(nn.Module):
    def forward(self, input):
        return input.view(32, -1)

class IMUNet(torch.nn.Module):
    
    def __init__(self, feature_extractor, num_features=128, dropout=0.5,
                 track_running_stats=False, pretrained=False):
        super(IMUNet, self).__init__()
        

        self.dropout = dropout
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        fc_in_features = self.feature_extractor.fc.in_features
        
        
        self.IMU_fc = nn.Sequential(
            Flatten(),
            nn.Linear(7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32),
            nn.ReLU(),
            FlattenIMU()
        )
        
        self.rnn = torch.nn.LSTM(
            input_size=6, 
            hidden_size= 64*2, 
            num_layers=4,
            batch_first=True)
        
        self.flat_imu = FlattenIMU()
        self.linear = nn.Linear(128,num_features)

        self.fc3 = torch.nn.Linear( 4096, num_features)
        self.feature_extractor.fc = torch.nn.Linear(fc_in_features, num_features)
        # Translation
        self.fc_xyz = torch.nn.Linear(num_features, 3)

        # Rotation in quaternions
        self.fc_quat = torch.nn.Linear(num_features, 4)
        
    def extract_features(self, image):
        x_features = self.feature_extractor(image)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
        return x_features
    
    def forward(self, data_input):
        image = data_input[0].cuda()
        imu_data = data_input[1].type(torch.FloatTensor).cuda()
#         print(imu_data.size()[1])
        #Images
        if type(image) is list:
            x_features = [self.extract_features(xi) for xi in image]
        elif torch.is_tensor(image):
            x_features = self.extract_features(image)
            
#         imu_features = self.IMU_fc(imu_data)
#         x = torch.cat((x_features, imu_features), dim=1).unsqueeze(0)
#         x = F.relu(self.rnn(x))
#         for i in range(imu_data.size()[0]):
        imu_feat = imu_data
#             for j in range(len(imu_feat)):
# #                 imu_features = self.IMU_fc(imu_feat[j])

#     #             for j in range(len(imu_features))
#                 print(imu_feat[j])
        r_out, (h_n, h_c) = self.rnn(imu_feat)
        imu_out = self.linear(r_out[:, -1, :]) # we want just the last time 


        imu_out = self.flat_imu(imu_out) # we want just the last time 
        

#         imu_out = self.linear(r_out[:, -1, :]) # we want just the last time 
#         print(x.shape)
#         imu_out = r_out

#         print(x_features.size())
#         print(imu_out.size())
        x = torch.cat((x_features, imu_out), dim=1)
#         print(x.size)
        x = F.relu(self.fc3(x))

        if type(x) is list:
#             x_features = [self.extract_features(xi) for xi in x]
            x_translations = [self.fc_xyz(xi) for xi in x]
            x_rotations = [self.fc_quat(xi) for xi in x]
            x_poses = [torch.cat((xt, xr), dim=1) for xt, xr in zip(x_translations, x_rotations)]  
        elif torch.is_tensor(x):
#             x_features = self.extract_features(x)
            x_translations = self.fc_xyz(x) 
            x_rotations = self.fc_quat(x)
#             print(x_translations.size())
#             print(x_rotations.size())
            x_poses = torch.cat((x_translations, x_rotations), dim=1)

        return x_poses