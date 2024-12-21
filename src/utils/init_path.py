import os
import glob

def init_path(checkpoint_dir, config_dir, size=512, old_version=False, preprocess='crop'):

    if old_version:
        #### load all the checkpoint of `pth`
        _3dtalker_paths = {
                'wav2lip_checkpoint' : os.path.join(checkpoint_dir, 'wav2lip.pth'),
                'audio2pose_checkpoint' : os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth'),
                'audio2exp_checkpoint' : os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth'),
                'free_view_checkpoint' : os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar'),
                'path_of_net_recon_model' : os.path.join(checkpoint_dir, 'epoch_20.pth')
        }

        use_safetensor = False
    elif len(glob.glob(os.path.join(checkpoint_dir, '*.safetensors'))):
        print('using safetensor as default')
        _3dtalker_paths = {
            "checkpoint":os.path.join(checkpoint_dir, '3DTalker_V0.0.2_'+str(size)+'.safetensors'),
            }
        use_safetensor = True
    else:
        print("WARNING: The new version of the model will be updated by safetensor, you may need to download it mannully. We run the old version of the checkpoint this time!")
        use_safetensor = False
        
        _3dtalker_paths = {
                        'wav2lip_checkpoint' : os.path.join(checkpoint_dir, 'wav2lip.pth'),
                        'audio2pose_checkpoint' : os.path.join(checkpoint_dir, 'auido2pose_00140-model.pth'),
                        'audio2exp_checkpoint' : os.path.join(checkpoint_dir, 'auido2exp_00300-model.pth'),
                        'free_view_checkpoint' : os.path.join(checkpoint_dir, 'facevid2vid_00189-model.pth.tar'),
                        'path_of_net_recon_model' : os.path.join(checkpoint_dir, 'epoch_20.pth')
                        }

    _3dtalker_paths['dir_of_BFM_fitting'] = os.path.join(config_dir) # , 'BFM_Fitting'
    _3dtalker_paths['audio2pose_yaml_path'] = os.path.join(config_dir, 'auido2pose.yaml')
    _3dtalker_paths['audio2exp_yaml_path'] = os.path.join(config_dir, 'auido2exp.yaml')
    _3dtalker_paths['use_safetensor'] =  use_safetensor # os.path.join(config_dir, 'auido2exp.yaml')
    if 'full' in preprocess:
        _3dtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00109-model.pth.tar')
        _3dtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender_still.yaml')
    else:
        _3dtalker_paths['mappingnet_checkpoint'] = os.path.join(checkpoint_dir, 'mapping_00229-model.pth.tar')
        _3dtalker_paths['facerender_yaml'] = os.path.join(config_dir, 'facerender.yaml')

    return _3dtalker_paths