import os
import argparse
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DATA
import imageio
from itertools import chain
from gan_loader import AMCDataset
from models.discogan import *
from runner.utils import torch_seed, get_config


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--noise', type=str, default='true', help='Set noise usage')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')

parser.add_argument('--gan_curriculum', type=int, default=10000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion( l2, torch.ones( l2.size() ).cuda() )
        losses += loss

    return losses

def get_gan_loss(label, dis_real, dis_fake, criterion, cuda):
    labels_dis_real = torch.ones([dis_real.size()[0], 1])
    labels_dis_fake = torch.zeros([dis_fake.size()[0], 1])
    labels_gen = torch.ones([dis_fake.size()[0], 1])

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion(dis_real, labels_dis_real) * 0.5 + criterion(dis_fake, labels_dis_fake) * 0.5
    gen_loss = criterion(dis_fake, labels_gen)

    return dis_loss, gen_loss


def main():

    global args
    args = parser.parse_args()
    cfg = get_config('./config.yaml')

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    epoch_size = 1000
    batch_size = 8
    learning_rate = 0.0002
    # win_size = 64
    # epoch_size = 1000
    # batch_size = 200
    # learning_rate = 0.0002


    img_result_path = './gan_results'
    model_path = './gan_checkpoint'
    
    if args.noise == 'true':
        img_result_path = os.path.join(img_result_path, 'noise')
        os.makedirs(img_result_path, exist_ok=True)
        model_path = os.path.join(model_path, 'noise')
        os.makedirs(model_path, exist_ok=True)  
    elif args.noise == 'false':
        img_result_path = os.path.join(img_result_path, 'original')
        os.makedirs(img_result_path, exist_ok=True)
        model_path = os.path.join(model_path, 'original')
        os.makedirs(model_path, exist_ok=True)

    train_dataset = None
    test_dataset = None

    train_dataset = AMCDataset(data_path='../amc_dataset/RML2018', 
                                mode='train',
                                snr_A=-20,
                                snr_B=20,
                                label='OOK')
    test_dataset = AMCDataset(data_path='../amc_dataset/RML2018', 
                                mode='test',
                                snr_A=-20,
                                snr_B=20,
                                label='OOK')

    generator_A = Generator()
    generator_B = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()
        discriminator_A = discriminator_A.cuda()
        discriminator_B = discriminator_B.cuda()

    data_size = train_dataset.__len__()

    train_dataloader = DATA.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DATA.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    recon_criterion = nn.MSELoss() # reconstruction
    gan_criterion = nn.BCELoss()   # GAN
    feat_criterion = nn.HingeEmbeddingLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    optim_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam(dis_params, lr=learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    
    gen_loss_total = []
    dis_loss_total = []

    # fix torch seed
    torch_seed(40)

    iters = 0
    # Train mode
    generator_A.train()
    generator_B.train()
    discriminator_A.train()
    discriminator_B.train()

    for epoch in range(epoch_size):
        print(f'==== Start epoch {epoch} ====')
        for sample_A, sample_B in tqdm.tqdm(train_dataloader):
            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            A = sample_A['data']
            B = sample_B['data']
            A_label = sample_A['label']
            B_label = sample_B['label']

            if cuda:
                A = A.cuda()
                B = B.cuda()

            A = A.float()
            B = B.float()

            AB = generator_B(A)
            BA = generator_A(B)

            ABA = generator_A(AB)
            BAB = generator_B(BA)

            # Reconstruction Loss
            recon_loss_A = recon_criterion(ABA, A)
            recon_loss_B = recon_criterion(BAB, B)

            # Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A(A)
            A_dis_fake, A_feats_fake = discriminator_A(BA)

            dis_loss_A, gen_loss_A = get_gan_loss(A_label, A_dis_real, A_dis_fake, gan_criterion, cuda)
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion)

            # Real/Fake GAN Loss (B)
            B_dis_real, B_feats_real = discriminator_B(B)
            B_dis_fake, B_feats_fake = discriminator_B(AB)

            dis_loss_B, gen_loss_B = get_gan_loss(B_label, B_dis_real, B_dis_fake, gan_criterion, cuda)
            fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion)

            # Total Loss

            if iters < args.gan_curriculum:
                rate = args.starting_rate
            else:
                rate = args.default_rate

            gen_loss_A_total = (gen_loss_B*0.1 + fm_loss_B*0.9) * (1.-rate) + recon_loss_A * rate
            gen_loss_B_total = (gen_loss_A*0.1 + fm_loss_A*0.9) * (1.-rate) + recon_loss_B * rate

            
            gen_loss = gen_loss_A_total + gen_loss_B_total
            dis_loss = dis_loss_A + dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            if iters % args.log_interval == 0:
                print("---------------------")
                print("GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean()))
                print("Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean()))
                print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
                print("DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean()))

            if iters % args.image_save_interval == 0:
                # Evaluation mode
                generator_A.eval()
                generator_B.eval()
                discriminator_A.eval()
                discriminator_B.eval()

                with torch.no_grad():
                    for test_A, test_B in tqdm.tqdm(test_dataloader):
                        if cuda:
                            test_A = test_A['data'].cuda()
                            test_B = test_B['data'].cuda()
                        
                        test_A = test_A.float()
                        test_B = test_B.float()

                        AB = generator_B(test_A)
                        BA = generator_A(test_B)
                        ABA = generator_A(AB)
                        BAB = generator_B(BA)

                        n_testset = min(test_A.size()[0], test_B.size()[0])

                        subdir_path = os.path.join(img_result_path, str(iters / args.image_save_interval))

                        if os.path.exists(subdir_path):
                            pass
                        else:
                            os.makedirs(subdir_path)

                        for im_idx in range(n_testset):
                            A_val = test_A[0].cpu().data.numpy().transpose(1,2,0) * 255.
                            B_val = test_B[0].cpu().data.numpy().transpose(1,2,0) * 255.
                            BA_val = BA[0].cpu().data.numpy().transpose(1,2,0) * 255.
                            ABA_val = ABA[0].cpu().data.numpy().transpose(1,2,0) * 255.
                            AB_val = AB[0].cpu().data.numpy().transpose(1,2,0) * 255.
                            BAB_val = BAB[0].cpu().data.numpy().transpose(1,2,0) * 255.

                            filename_prefix = os.path.join (subdir_path, str(im_idx))
                            imageio.imwrite( filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:,:,::-1])
                            imageio.imwrite( filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:,:,::-1])
                            imageio.imwrite( filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:,:,::-1])
                            imageio.imwrite( filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:,:,::-1])
                            imageio.imwrite( filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:,:,::-1])
                            imageio.imwrite( filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:,:,::-1])

            if iters % args.model_save_interval == 0:
                torch.save( generator_A, os.path.join(model_path, 'model_gen_A-' + str( iters / args.model_save_interval )))
                torch.save( generator_B, os.path.join(model_path, 'model_gen_B-' + str( iters / args.model_save_interval )))
                torch.save( discriminator_A, os.path.join(model_path, 'model_dis_A-' + str( iters / args.model_save_interval )))
                torch.save( discriminator_B, os.path.join(model_path, 'model_dis_B-' + str( iters / args.model_save_interval )))

            iters += 1
    

if __name__=="__main__":
    main()