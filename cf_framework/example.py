
    
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torchvision.transforms import v2
from utils import create_recons_models
from utils.distil import accuracy, ExpectedCalibrationError, uncertainty, custom_output
from utils.gan import hinge_d_loss, vanilla_d_loss, default_descriminator_config
from utils.discriminator.models import NLayerDiscriminator, weights_init, DenseDescriminator
from utils.pack import PivoitDataset
import matplotlib.pyplot as plt  

class Regular(L.LightningModule):
    def __init__(self, model, desc_config=default_descriminator_config(), disc_loss="hinge", n_classes=100, fmp_size=(7, 7), device='cuda:0', 
                 warm_up={'same_loss': 1000, 'cls_loss': 0}, 
                 weight={'same_loss': 1, 'cls_loss': 1}, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.warm_up = warm_up
        self.weight = weight
        self.ece_metric = ExpectedCalibrationError(n_bin=5, device=device)
        if desc_config['type'] == 'Patch':
            self.discriminator = NLayerDiscriminator(**desc_config['param']).apply(weights_init)
        elif desc_config['type'] == 'Dense':
            self.discriminator = DenseDescriminator(**desc_config['param']).apply(weights_init)
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"Discriminator running with {disc_loss} loss.")
        self.fake_history = None#使用生成器历史样本防止判别器震荡
        self.fake_cond_history = None#历史负样本的类别
        self.real_history = None
        self.real_cond_history = None
        self.history_length = 100
        self.automatic_optimization = False#关闭自动优化

        self.n_classes = n_classes
        self.fmp_size = fmp_size
        #将one-hot条件转化为目标类别的条件嵌入
        self.cls_condition = nn.Embedding(n_classes, fmp_size[0]*fmp_size[1])#添加一层特征图作为每个样本的类别条件，判别器需要在指定条件先判别正负样本

    def get_condition(self, cond):
        return self.cls_condition(cond).reshape((-1, self.fmp_size[0], self.fmp_size[1])).unsqueeze(1)
    
    def get_param_group(self):
        return {'param_G': list(self.model.parameters()), 
                'param_D': list(self.discriminator.parameters()) + list(self.cls_condition.parameters())}

    def training_step(self, batch, batch_idx):
        # if self.global_step // len(self.optimizers()) % 2000 == 0:#每2k步重新进行一次划分
        #     self.pivoit_method(p=0.8)
        #     self.fake_history = None

        x, y, use_label = batch
        features = self.model.forward_features(x)#[B, C, H, W]
        logit = self.model.forward_head(features)#[B, N]
        ood_score = uncertainty(logit, T=1)
        ood_mask = ood_score < -4.622#所有样本的分布置信度
        # pos_mask = use_label & ood_mask
        # neg_mask = ~use_label & ood_mask#负样本中置信的部分
        
        feature_sup = features[use_label]
        feature_sup = feature_sup if feature_sup.numel() > 0 else None
        cond_sup = y[use_label]
        feature_unsup = features[~use_label]
        feature_unsup = feature_unsup if feature_unsup.numel() > 0 else None
        cond_unsup = y[~use_label]

        # same_feature = features[neg_mask]
        # same_cond = y[neg_mask]

        # logit_sup = logit[use_label]
        # label_sup = y[use_label]

        # feature_sup = None
        # feature_unsup = None
        # cond_sup = None
        # cond_unsup = None
        # logit_sup = None
        # label_sup = []
        # for i, sup in enumerate(use_label):#根据数据集划分对特征和输出进行分割
        #     if sup:
        #         # feature_sup = torch.cat((feature_sup, features[i].unsqueeze(0)), dim=0) if feature_sup is not None else features[i].unsqueeze(0)
        #         # cond_sup = torch.cat((cond_sup, y[i].unsqueeze(0)), dim=0) if cond_sup is not None else y[i].unsqueeze(0)
        #         # logit_sup = torch.cat((logit_sup, logit[i].unsqueeze(0)), dim=0) if logit_sup is not None else logit[i].unsqueeze(0)
        #         # label_sup.append(y[i])
        #     else:
        #         feature_unsup = torch.cat((feature_unsup, features[i].unsqueeze(0)), dim=0) if feature_unsup is not None else features[i].unsqueeze(0)
        #         cond_unsup = torch.cat((cond_unsup, y[i].unsqueeze(0)), dim=0) if cond_unsup is not None else y[i].unsqueeze(0)
        
        #特征一致性损失
        if feature_unsup is not None:
            # loss = self.discriminator(torch.cat((features, self.get_condition(y).detach()), dim=1))#[B, 1]
            loss_same = -self.discriminator(torch.cat((feature_unsup, self.get_condition(cond_unsup).detach()), dim=1)).mean()#检测无监督部分的特征与监督部分的区别
            # p_sup = self.discriminator(torch.cat((feature_sup, self.get_condition(cond_sup).detach()), dim=1)).mean()
            # loss_same = loss[use_label].mean() - loss[~use_label].mean()
        else:
            loss_same = 0
        #有监督损失
        # loss_cls = self.loss_fn(logit_sup, torch.tensor(label_sup, device=self.device))#监督部分分类损失
        loss_cls = self.loss_fn(logit, y)#使用全部监督信号
        fake_feature = self.get_samples(feature_unsup, cond_unsup, types='fake')
        real_feature = self.get_samples(feature_sup, cond_sup, types='real')
        # fake_feature = torch.cat((feature_unsup.detach(), self.get_condition(cond_unsup)), dim=1)
        if real_feature is not None and fake_feature is not None:
            logits_fake = self.discriminator(fake_feature)
            logits_real = self.discriminator(real_feature)
            #判别器损失
            loss_d = self.disc_loss(logits_real, logits_fake)
        else:
            loss_d = -1
        self.update_history(feature_unsup, cond_unsup, types='fake')
        self.update_history(feature_sup, cond_sup, types='real')
        #获取所有loss的权重
        same_loss_weight = self.weight['same_loss'] if self.global_step // len(self.optimizers()) >= self.warm_up['same_loss'] else 0
        cls_loss_weight = self.weight['cls_loss'] if self.global_step // len(self.optimizers()) >= self.warm_up['cls_loss'] else 0
        # desc_loss_weight = self.weight['desc_loss'] if self.global_step // len(self.optimizers()) >= self.warm_up['desc_loss'] else 0
        loss_aux = loss_same * same_loss_weight + loss_cls * cls_loss_weight
        # loss_d = loss_d * desc_loss_weight
        opt_G, opt_D = self.optimizers()
        #先更新生成器（生成器会对判别器产生梯度）
        opt_G.zero_grad()
        self.manual_backward(loss_aux)
        opt_G.step()
        #再更新判别器（判别器不会对生成器产生梯度）
        opt_D.zero_grad()
        if loss_d != -1:
            self.manual_backward(loss_d)
            opt_D.step()
        self.log_dict({'same_loss': loss_same, 'cls_loss': loss_cls, 'desc_loss': loss_d, 'select_rate': ood_mask.sum()/ood_mask.numel()})

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        acc = accuracy(pred, y, topk=(1, 5))
        self.ece_metric.update(y, pred)
        # return {'val_loss': loss, 'top1_acc': acc[0], 'top5_acc': acc[1]}
        self.log_dict({'val_loss': loss})
        self.log_dict({'top1_acc': acc[0]})
        self.log_dict({'top5_acc': acc[1]})

    def on_validation_epoch_end(self): 
        ece_loss = self.ece_metric.get_loss()
        self.log_dict({'ECE': ece_loss})
        ece_values = self.ece_metric.get_statistic().cpu().numpy()  
        # 设置柱状图的 x 轴标签  
        x_labels = [f'Confidence {(i+1)*self.ece_metric.interval:.2f}' for i in range(len(self.ece_metric))]  
        # 绘制柱状图  
        plt.bar(x_labels, ece_values, color='blue')  
        plt.xlabel('Segments')  
        plt.ylabel('Accuracy')  
        plt.title(f'ECE Metrics {ece_loss:.2f}')  
        plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好显示  
        plt.tight_layout()  # 自适应布局  
        plt.savefig(f'ECE on {self.global_step//len(self.optimizers())} step', dpi=300)  
        plt.close()
        self.ece_metric.reset()
    
    def configure_optimizers(self):
        gan_params = self.get_param_group()
        param_group_G = gan_params['param_G']
        param_group_D = gan_params['param_D']
        optimizer_G = torch.optim.Adam(param_group_G, lr=1e-3, betas=[0.5, 0.9])#更新生成器
        optimizer_D = torch.optim.Adam(param_group_D, lr=1e-3, betas=[0.5, 0.9])#更新判别器
        return optimizer_G, optimizer_D
    
    def update_history(self, new_sample, new_cond, types='fake'):
        history = self.fake_history if types == 'fake' else self.real_history
        cond_history = self.fake_cond_history if types == 'fake' else self.real_cond_history
        if new_sample is None:#当前样本是空
            return
        if history is None:
            history = new_sample.detach()
            cond_history = new_cond.detach()
        else:
            idx = min(history.shape[0], self.history_length-len(new_sample))#维持队列大小条件下能保留前xx个样本
            history = torch.cat((history[-idx:], new_sample.detach()), dim=0)#添加新样本
            cond_history = torch.cat((cond_history[-idx:], new_cond.detach()), dim=0)
        if types == 'fake':
            self.fake_history = history
            self.fake_cond_history = cond_history
        else:
            self.real_history = history
            self.real_cond_history = cond_history

    def get_samples(self, new_sample, new_cond, types='fake'):
        history = self.fake_history if types == 'fake' else self.real_history
        cond_history = self.fake_cond_history if types == 'fake' else self.real_cond_history
        if new_sample is None:#当前样本不存在就返回历史负样本
            if history is not None:
                cond = self.get_condition(cond_history)
                history = torch.cat((history, cond), dim=1)
            return history
        elif history is None:#当前样本存在历史样本不存在就返回当前样本
            cond = self.get_condition(new_cond)
            return torch.cat((new_sample.detach(), cond), dim=1)
        else:#两者皆存在
            cond = torch.cat((cond_history, new_cond), dim=0)#添加历史样本条件
            history = torch.cat((history, new_sample.detach()), dim=0)#添加历史负样本
            return torch.cat((history, self.get_condition(cond)), dim=1)#合并条件到特征中
        

cfg = {
    'resnet18': {
        'model': 'resnet18.a1_in1k',
        'ckpt': '/root/autodl-tmp/Lab/timmModels/resnet18/resnet18.fb_swsl_ig1b_ft_in1k.bin',
        'kwargs': {
            'num_classes': 100, 
            # 'strict': False,
        },
    }
}

desc_cfg = {
        'type': 'Patch',
        'param': {
            'input_nc': 513,
            'n_layers': 1,
            'use_actnorm': False,
            'ndf': 64,
        }
}

bz = 64
device = 'cuda:0'
train_dataset_path = '/root/autodl-tmp/Lab/datasets/flowers102/train'
val_dataset_path = '/root/autodl-tmp/Lab/datasets/flowers102/val'

train_dataset_path = '/root/autodl-fs/datasets/ImageNet100subset/train'
val_dataset_path = '/root/autodl-fs/datasets/ImageNet100subset/val'

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # 监视指标为验证集loss
    dirpath='./checkpoints',
    filename='model-{epoch:02d}-{top1_acc:.2f}',
    save_top_k=1,  # 只保存验证集loss最低的1个模型
    mode='min'  # 以最小验证集loss为目标
)

if __name__ =='__main__':
    models = create_recons_models(cfg)
    for t in models:
        cls_model, train_transform = models[t]
    cls_model.to(device)
    model = Regular(model=cls_model, weight={'same_loss': 0.1, 'cls_loss': 1}, warm_up={'same_loss': 16000, 'cls_loss': 0}, n_classes=100, fmp_size=(7, 7), desc_config=desc_cfg)
    val_tansform = train_transform
    train_transform = v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.RandomHorizontalFlip(),         # 随机水平翻转  
        v2.RandomRotation(30),             # 随机旋转，范围为-30到30度  
        v2.ColorJitter(brightness=0.2,    # 随机调整亮度  
                            contrast=0.2,      # 随机调整对比度  
                            saturation=0.2,    # 随机调整饱和度  
                            hue=0.1),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=([0.4850, 0.4560, 0.4060]), std=([0.2290, 0.2240, 0.2250]))
    ])
    #original version
    train_dataset = tv.datasets.ImageFolder(root=train_dataset_path, transform=train_transform)
    pivoit_dataset = PivoitDataset(datasets=train_dataset, p=0.2, n_classes=100)
    train_iter = data.DataLoader(pivoit_dataset, batch_size=bz, shuffle=True, num_workers=16)
    val_dataset = tv.datasets.ImageFolder(root=val_dataset_path, transform=val_tansform)
    test_iter = data.DataLoader(val_dataset, batch_size=bz, shuffle=True, num_workers=16)

    logger = WandbLogger(project='IM100pretrain')
    # logger = CSVLogger("LocalExp", name="test")
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=100, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_iter, test_iter)

    '''for test '''
    # model = custom_output(model=cls_model, target_layer=['layer2'])
    # output = model(torch.randn((5, 3, 224, 224), device=device))
    # for k in output:
    #     print(f'{k}:, {output[k][-1].shape}')
    # for c in p_val.cls_ids:
    #     print(f'class {c} has samples:{len(p_val.cls_ids[c])}')
    # x = torch.randn((1, 3, 224, 224), device=device)
    # y, intermedia = cls_model.forward_intermediates(x)
    # print(f'y shape:{y.shape}')
    # for i in intermedia:
    #     print(f'feature_shape:{i.shape}')