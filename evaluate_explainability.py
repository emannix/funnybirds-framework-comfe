import argparse
import random

import torch
# from captum.attr import IntegratedGradients, InputXGradient

# from models.resnet import resnet50
# from models.vgg import vgg16
# from models.ViT.ViT_new import vit_base_patch16_224
# from models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
# from models.model_wrapper import StandardModel, ViTModel
from evaluation_protocols import accuracy_protocol, controlled_synthetic_data_check_protocol, single_deletion_protocol, preservation_check_protocol, deletion_check_protocol, target_sensitivity_protocol, distractibility_protocol, background_independence_protocol
# from explainers.explainer_wrapper import CaptumAttributionExplainer, ViTGradCamExplainer, ViTRolloutExplainer, ViTCheferLRPExplainer, CustomExplainer

from explainers.explainer_wrapper import ComFeExplainer
from models.model_wrapper import ComFeModel

parser = argparse.ArgumentParser(description='FunnyBirds - Explanation Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--model', required=True,
                    help='model architecture')
parser.add_argument('--explainer', required=True,
                    help='explainer')
parser.add_argument('--checkpoint_name', type=str, required=False, default=None,
                    help='checkpoint name (including dir)')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='seed')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
parser.add_argument('--nr_itrs', default=2501, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
                    
parser.add_argument('--accuracy', default=False, action='store_true',
                    help='compute accuracy')
parser.add_argument('--controlled_synthetic_data_check', default=False, action='store_true',
                    help='compute controlled synthetic data check')
parser.add_argument('--single_deletion', default=False, action='store_true',
                    help='compute single deletion')
parser.add_argument('--preservation_check', default=False, action='store_true',
                    help='compute preservation check')
parser.add_argument('--deletion_check', default=False, action='store_true',
                    help='compute deletion check')
parser.add_argument('--target_sensitivity', default=False, action='store_true',
                    help='compute target sensitivity')
parser.add_argument('--distractibility', default=False, action='store_true',
                    help='compute distractibility')
parser.add_argument('--background_independence', default=False, action='store_true',
                    help='compute background dependence')


# dinov2 wreg
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-02-07_10-12_36f53_final_plainseg_interpretable2_global_run_funnybirds_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_3 v_interpretable/

# dinov2
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-01_21-19_5b4d5_final_plainseg_interpretable2_global_run_funnybirds_noreg_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_5 v_interpretable/

# DINO
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-01_21-19_46be7_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_224.d.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTru_1_1 v_interpretable/

# CLIP
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-01_21-19_46be7_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_c_224.o.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pT_0_4 v_interpretable/

# augreg
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-01_23-48_18747_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_224.a_in21.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_2_3 v_interpretable/

# MAE
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-01_21-19_46be7_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_224.m.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTru_3_4 v_interpretable/


# dinov2-vits
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-03_11-19_7089f_final_plainseg_interpretable2_global_run_funnybirds_noreg_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_2 v_interpretable/

# dinov2 wreg-vits
# rsync -avzh   \
#   nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/lightly-wrapper/output/local/2025-03-03_11-19_3575d_final_plainseg_interpretable2_global_run_funnybirds_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_4 v_interpretable/


def main():
    args = parser.parse_args()
    device = 'cuda:' + str(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create model
    if args.model == 'resnet50':
        model = resnet50(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vgg16':
        model = vgg16(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vit_b_16':
        if args.explainer == 'CheferLRP':
            model = vit_LRP(num_classes=50)
        else:
            model = vit_base_patch16_224(num_classes = 50)
        model = ViTModel(model)
    elif 'comfe' in args.model:
        import sys
        sys.path.append('/home/unimelb.edu.au/nbloomfield/OneDrive/PhD/Repositories/lightly-wrapper/lightly_play')
        sys.path.append('/data/cephfs/punim0980/lightly-wrapper/lightly_play')
        from goo.methods.interpretable.prototypes_supervised2_global2 import TSNE as ComFe
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        model_dir = 'v_interpretable'
        model_dir = 'output/local'

        if args.explainer == 'comfe_dinov2_wreg':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-02-07_10-12_36f53_final_plainseg_interpretable2_global_run_funnybirds_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_3/checkpoints/epoch_037.ckpt'
        elif args.explainer == 'comfe_dinov2':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-01_21-19_5b4d5_final_plainseg_interpretable2_global_run_funnybirds_noreg_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_5/checkpoints/epoch_035.ckpt'
        elif args.explainer == 'comfe_dino':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-01_21-19_46be7_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_224.d.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTru_1_1/checkpoints/epoch_044.ckpt'
        elif args.explainer == 'comfe_clip':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-01_21-19_46be7_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_c_224.o.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pT_0_4/checkpoints/epoch_048.ckpt'
        elif args.explainer == 'comfe_augreg':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-01_23-48_18747_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_224.a_in21.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_2_3/checkpoints/epoch_047.ckpt'
        elif args.explainer == 'comfe_mae':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-01_21-19_46be7_final_plainseg_interpretable2_global_run_funnybirds_otherbackbones_1-6_d_o.t_f_e_p.n.n.b.f_m_b_ph16_224.m.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTru_3_4/checkpoints/epoch_035.ckpt'
        elif args.explainer == 'comfe_dinov2-vits':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-03_11-19_7089f_final_plainseg_interpretable2_global_run_funnybirds_noreg_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_2/checkpoints/epoch_049.ckpt'
        elif args.explainer == 'comfe_dinov2_wreg-vits':
            model_path = '../../lightly-wrapper/'+model_dir+'/2025-03-03_11-19_3575d_final_plainseg_interpretable2_global_run_funnybirds_1-6_d_o.t_f_e_p.n.n.b_c.n_cs50.n.n.b_c.n_c_ps150.n.n.b_c.l_pTrue.n.n.b_c.l_p_pTrue.n.n.b_c.l_p_cTrue.n.n_0_4/checkpoints/epoch_029.ckpt'

        comfe_model = ComFe.load_from_checkpoint(model_path)

        explainer = ComFeExplainer(comfe_model)
        model = ComFeModel(comfe_model, explainer)
    else:
        print('Model not implemented')
    
    if args.checkpoint_name:
        model.load_state_dict(torch.load(args.checkpoint_name, map_location=torch.device('cpu'))['state_dict'])
    model = model.to(device)
    model.eval()

    # create explainer
    if args.explainer == 'InputXGradient':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == 'IntegratedGradients':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,256,256)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif args.explainer == 'Rollout':
        explainer = ViTRolloutExplainer(model)
    elif args.explainer == 'CheferLRP':
        explainer = ViTCheferLRPExplainer(model)
    elif 'comfe' in args.explainer:
        explainer = ComFeExplainer(comfe_model)
    elif args.explainer == 'CustomExplainer':
        ...
    else:
        print('Explainer not implemented')

    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        print('Computing accuracy...')
        accuracy = accuracy_protocol(model, args)
        accuracy = round(accuracy, 5)

    if args.controlled_synthetic_data_check:
        print('Computing controlled synthetic data check...')
        csdc = controlled_synthetic_data_check_protocol(model, explainer, args)

    if args.target_sensitivity:
        print('Computing target sensitivity...')
        ts = target_sensitivity_protocol(model, explainer, args)
        ts = round(ts, 5)

    if args.single_deletion:
        print('Computing single deletion...')
        sd = single_deletion_protocol(model, explainer, args)
        sd = round(sd, 5)

    if args.preservation_check:
        print('Computing preservation check...')
        pc = preservation_check_protocol(model, explainer, args)

    if args.deletion_check:
        print('Computing deletion check...')
        dc = deletion_check_protocol(model, explainer, args)

    if args.distractibility:
        print('Computing distractibility...')
        distractibility = distractibility_protocol(model, explainer, args)

    if args.background_independence:
        print('Computing background independence...')
        background_independence = background_independence_protocol(model, args)
        background_independence = round(background_independence, 5)
    
    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = csdc[threshold]/3. + pc[threshold]/3. + dc[threshold]/3. + distractibility[threshold]
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print('FINAL RESULTS:')
    print('Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS')
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(accuracy, round(csdc[best_threshold],5), round(pc[best_threshold],5), round(dc[best_threshold],5), round(distractibility[best_threshold],5), background_independence, sd, ts))
    print('Best threshold:', best_threshold)

if __name__ == '__main__':
    main()