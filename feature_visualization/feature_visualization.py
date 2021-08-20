import torch

from data.dataset import ModelNetDataLoader
from tqdm import tqdm
from models.model import PointTransformerCls
import numpy as np

from utils import provider
from utils.feature_extractor import FeatureExtractor
from config import config
import logging
import torch.nn as nn
import os
from utils.utils import calculate_accuracy


class FeatureVisualization:
    def __init__(self):
        self.config = config.load_config('../config/config.ini')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = self.config['DataPrep']['dataset']
        self.npoints = int(self.config['Net']['num_points'])
        self.output_path = os.path.join(self.config['Runtime']['output_path'], self.name + '.npz')
        self.num_classes = int(self.config['DataPrep']['num_classes'])

        logging_path = os.path.join(self.config['Runtime']['output_path'], self.name + '.log')
        logging.basicConfig(filename=logging_path)
        self.logger = logging.getLogger('compute %s:' % self.name)

        self.original_model_path = os.path.join(self.config['Net']['original_model_path'],'best_model.pth')
        self.model_path = os.path.join(self.config['Net']['model_path'],'best_model.pth')
        self.testDataLoader = self.load_test_data()
        self.valDataLoader, self.trainDataLoader = self.load_train_data()

        self.PCT_model = PointTransformerCls().to(self.device)
        self.extractor = self.load_original_model(self.original_model_path)
        self.classifier = self.load_new_model()

        self.start_epoch, self.best_instance_acc = self.load_checkpoint()
        self.weights = []

    def load_test_data(self):
        TEST_DATASET = ModelNetDataLoader(root=self.data_path, npoint=self.npoints, split='test', normal_channel=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)
        return testDataLoader

    def load_train_data(self):
        batch_size = int(self.config['DataPrep']['batch_size'])
        FULL_DATASET = ModelNetDataLoader(root=self.data_path, npoint=self.npoints, split='train',
                                          normal_channel=True)
        torch.manual_seed(0)
        trainDataSet, valDataSet = torch.utils.data.random_split(FULL_DATASET, [int(0.8 * len(FULL_DATASET)),
                                                                                len(FULL_DATASET) - int(
                                                                                    0.8 * len(FULL_DATASET))])
        trainDataLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=batch_size, shuffle=True,
                                                      num_workers=4)
        valDataLoader = torch.utils.data.DataLoader(valDataSet, batch_size=batch_size, shuffle=True,
                                                    num_workers=4)

        return valDataLoader, trainDataLoader

    def load_original_model(self, model_path):
        PCT_checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.PCT_model.load_state_dict(PCT_checkpoint['model_state_dict'])
        extractor = FeatureExtractor(self.PCT_model.eval())
        return extractor

    def load_new_model(self, model_path):
        raise Exception("Not Implemented")

    def calculate_features(self, features, weights, target):
        raise Exception("Not Implemented")

    def eval(self):
        self.classifier.eval()
        pred_array = []
        target_array = []
        maps_array = []
        pts_array = []
        for j, data in tqdm(enumerate(self.testDataLoader), total=len(self.testDataLoader)):
            points, target = data
            target = target[:, 0]
            points, target = points.to(self.device), target.to(self.device)

            self.extractor.extract(points)
            F, pred = self.classifier(self.extractor.conv_fuse, train=False)
            maps = self.calculate_features(F.cpu().detach().numpy(), self.weights, target)

            maps_array.append(maps)
            pts_array.append(points.detach().cpu().numpy())
            pred_array.append(pred.detach().cpu().numpy())
            target_array.append(target.detach().cpu().numpy())

        np.savez(self.output_path, pred=pred_array, target=target_array, points=pts_array,
                 maps=maps_array)  # save all in one file

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=torch.device(self.device))
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_instance_acc = checkpoint['instance_acc']
        except:
            self.logger.info('No existing model, starting training from scratch...')
            start_epoch = 0
            best_instance_acc = 0.0
        return start_epoch,best_instance_acc

    def train(self):

        for param in self.PCT_model.parameters():
            param.requires_grad_(False)

        lr = float(self.config['Trainer']['learning_rate'])
        weight_decay = float(self.config['Trainer']['weight_decay'])
        epochs = int(self.config['Trainer']['epochs'])

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

        global_epoch = 0
        global_step = 0
        mean_correct = []

        '''TRANING'''
        self.logger.info('Start training...')
        for epoch in range(self.start_epoch, epochs):
            self.logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, epochs))

            self.classifier.train()
            for batch_id, data in tqdm(enumerate(self.trainDataLoader, 0), total=len(self.trainDataLoader),
                                       smoothing=0.9):
                points, target = data
                points = points.data.numpy()
                points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                target = target[:, 0]
                points, target = points.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                self.extractor.extract(points)
                pred = self.classifier(self.extractor.conv_fuse)
                loss = criterion(pred, target.long())
                mean_correct.append(calculate_accuracy(pred, target) / float(points.size()[0]))

                loss.backward()
                optimizer.step()
                global_step += 1

            scheduler.step()

            train_instance_acc = np.mean(mean_correct)
            if train_instance_acc >= self.best_instance_acc:
                self.logger.info('Best Instance Accuracy: %f' % train_instance_acc)
                self.best_instance_acc = train_instance_acc
                best_epoch = epoch + 1
                self.logger.info('Save model...')
                self.logger.info('Saving at %s' % self.model_path)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': train_instance_acc,
                    'model_state_dict': self.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, self.model_path)

            with torch.no_grad():
                instance_acc = self.test(self.classifier.eval(), self.valDataLoader)
                self.logger.info('Validation Instance Accuracy: %f' % (instance_acc))
                global_epoch += 1

        self.logger.info('End of training...')

    def test(self, model, loader):
        mean_correct = []
        for j, data in tqdm(enumerate(loader), total=len(loader)):
            points, target = data
            target = target[:, 0]
            points, target = points.cuda(), target.cuda()
            classifier = model.eval()

            self.extractor.extract(points)
            pred = classifier(self.extractor.conv_fuse)
            mean_correct.append(calculate_accuracy(pred, target) / float(points.size()[0]))
        instance_acc = np.mean(mean_correct)
        return instance_acc
