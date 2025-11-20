import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    
 
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])



    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(   # 预先定义一下 a beta 等参数
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                image_name = train_data[1][0]
                train_data = train_data[0]
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    # result_path_hr = result_path + '/hr'
                    # os.makedirs(result_path_hr, exist_ok=True)
                    # print(result_path,result_path_hr)
                    # exit()

                    result_path_hr = result_path + '/'+str(current_epoch) + '/'+'hr'
                    os.makedirs(result_path_hr, exist_ok=True)
                    result_path_sr = result_path + '/'+str(current_epoch) + '/'+'sr'
                    os.makedirs(result_path_sr, exist_ok=True)
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        image_name = val_data[1][0]
                        val_data = val_data[0]
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)  
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint16
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint16
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint16
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint16

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/Ground.png'.format(result_path_hr, image_name))
                        Metrics.save_img(
                            sr_img, '{}/Restore.png'.format(result_path_sr, image_name))
                        # Metrics.save_img(
                        #     hr_img, '{}/{}_hr.png'.format(result_path, image_name))
                        # Metrics.save_img(
                        #     sr_img, '{}/{}_sr.png'.format(result_path, image_name))
                        # Metrics.save_img(
                        #     lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(
                        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        # if wandb_logger:
                        #     wandb_logger.log_image(
                        #         f'validation_{idx}', 
                        #         np.concatenate((fake_img, sr_img, hr_img), axis=1)
                        #     )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.!!!')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        
        result_path_hr = result_path + '/hr/'
        os.makedirs(result_path_hr, exist_ok=True)
        result_path_sr = result_path + '/sr/'
        os.makedirs(result_path_sr, exist_ok=True)
        all_times = [] 
        import time 
        for _,val_data in enumerate(val_loader):
            idx += 1
            image_name = val_data[1][0]
            val_data = val_data[0]

            diffusion.feed_data(val_data)
            start_time = time.time()
            diffusion.test(continous=False)
            end_time = time.time()
            all_times.append(end_time-start_time)
            visuals = diffusion.get_current_visuals()
            hr_img = Metrics.tensor2img(visuals['HR'])  
            lr_img = Metrics.tensor2img(visuals['LR'])  
            fake_img = Metrics.tensor2img(visuals['INF'])  


            sr_img = visuals['SR']  # uint8
            hr_image = visuals['HR']
            
            new_result_path_sr = result_path_sr.replace("\\",'/')
            new_result_path_hr = result_path_hr.replace("\\",'/')
            image_name = image_name.split('\\')[-1]
            Metrics.save_img(
                Metrics.tensor2img(sr_img), '{}/{}.png'.format(new_result_path_sr, image_name))
            Metrics.save_img(
                Metrics.tensor2img(hr_image), '{}/{}.png'.format(new_result_path_hr, image_name))
            
            # generation
            # print(visuals['SR'].shape)
            # print(Metrics.tensor2img(visuals['SR']).shape)
            # eval_psnr = Metrics.calculate_psnr(visuals['SR'][-1], hr_img)
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR']), hr_img)
            # eval_ssim = Metrics.calculate_ssim(visuals['SR'][-1], hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR']), hr_img)
            
            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)
        print('测试集中平均单张图像的测试时间：', sum(all_times)/len(all_times))
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        
        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })



# train : python sr.py -p train -c config/sr_sr3_16_128.json

# test : python sr.py -p val -c config/sr_sr3_16_128.json

# pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple



# nohup python sr.py -p train -c config/sr_sr3_16_128.json > output_train.log 2>&1 &  [1] 36731   tail -f output_train.log
# nohup python sr.py -p val -c config/sr_sr3_16_128.json > output_test.log 2>&1 &            56323     tail -f output_test.log
# 