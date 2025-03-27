import warnings
import argparse
import pickle
import yaml
from omegaconf import OmegaConf

import time

import torch
import pytorch_lightning as pl
import itertools
import networkx as nx
torch.cuda.empty_cache()

from omegaconf import DictConfig
from torch_geometric.utils import to_dense_adj
from tqdm.auto import tqdm
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization, NonMolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
# from src.datasets.dataset_utils import to_nx
from src.reduction.reduction import get_comm_coarsed_graph
# from src.coarsed_diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.coarse_diffusion_model_discrete import CoarseDiscreteDenoisingDiffusion
from THGD.src.refinement_diffusion_model_discrete import ExpandDiscreteDenoisingDiffusion
from src.guidance.masked_guidance_diffusion_model_discrete import MaskedGuidanceDiscreteDenoisingDiffusion
from src.guidance.masked_regressor_discrete import MaskedRegressorDiscrete
from src.datasets.dataset_utils import expand_graphs, repeat_batch, coarsed_sample_loader, expanded_sample_loader

warnings.filterwarnings("ignore", category=PossibleUserWarning)

def load_coarsed_model(global_cfg, cfg, resume):
    dataset_config = cfg.dataset
    if dataset_config['name'] == 'coarsed_mose':
        from src.datasets import mose_dataset
        datamodule = mose_dataset.MoseDataModule(cfg)
        dataset_infos = mose_dataset.CoarseMoseInfos(cfg)
    if dataset_config['name'] == 'coarsed_guacamol':
        from src.datasets import guacamol_dataset
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.CoarseGuacamolInfos(cfg)
    if dataset_config['name'] == 'coarsed_zinc250k':
        from src.datasets import zinc_dataset
        datamodule = zinc_dataset.ZINC250kDataModule(cfg)
        dataset_infos = zinc_dataset.CoarseZINC250kInfos(cfg)


    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)
    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = None
    visualization_tools = NonMolecularVisualization()
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features, 
                        }
    model = CoarseDiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs, strict=False)

    OmegaConf.set_struct(model.cfg.general, False)
    model.cfg.general.samples_to_save = global_cfg.save_final
    model.cfg.general.chains_to_save = global_cfg.keep_chain
    model.cfg.general.number_chain_steps = global_cfg.number_chain_steps
    model.cfg.general.resample_step = global_cfg.resample_step
    model.cfg.train.batch_size = global_cfg.batch_size
    OmegaConf.set_struct(model.cfg.general, True)
    return model, dataset_infos

def load_expanded_model(global_cfg, cfg, resume):
    dataset_config = cfg["dataset"]
    if dataset_config['name'] == 'expanded_mose':
        from src.datasets import mose_dataset
        datamodule = mose_dataset.MoseDataModule(cfg)
        dataset_infos = mose_dataset.ExpandMoseInfos(cfg)

    if dataset_config['name'] == 'expanded_guacamol':
        from src.datasets import guacamol_dataset
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.ExpandGuacamolInfos(cfg)

    if dataset_config['name'] == 'expanded_zinc250k':
        from src.datasets import zinc_dataset
        datamodule = zinc_dataset.ZINC250kDataModule(cfg)
        dataset_infos = zinc_dataset.ExpandZINC250kInfos(cfg)

    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)
    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, None)
    visualization_tools = MolecularVisualization(None, dataset_infos=dataset_infos)
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features, 
                        }
    model = ExpandDiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs, strict=False)

    OmegaConf.set_struct(model.cfg.general, False)
    model.cfg.general.samples_to_save = global_cfg.save_final
    model.cfg.general.chains_to_save = global_cfg.keep_chain
    model.cfg.general.number_chain_steps = global_cfg.number_chain_steps
    model.cfg.general.resample_step = global_cfg.resample_step
    model.cfg.general.skip = global_cfg.skip_t
    model.cfg.general.disable_tqdm = not global_cfg.enable_progress_bar
    model.cfg.train.batch_size = global_cfg.batch_size
    OmegaConf.set_struct(model.cfg.general, True)
    return model, datamodule, dataset_infos, model_kwargs

def sample_coarse_graph(coarsed_model, devices, coarsed_loader,
                        enable_progress_bar=True, 
                        save_graphs=True):
    
    coarsed_trainer = pl.Trainer(
                    strategy="ddp",  # Needed to load old checkpoints
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=devices,
                    enable_progress_bar=enable_progress_bar)
    

    start = time.time()
    coarsed_graphs = []
    _coarsed_graphs = coarsed_trainer.predict(coarsed_model, dataloaders=coarsed_loader)
    for graphs in _coarsed_graphs:
        coarsed_graphs.extend(graphs)

    print(f'Sampling Coarsed Graphs took {time.time() - start:.2f} seconds\n')

    if save_graphs:
        with open('./coarsed_graphs.pkl', 'wb') as f:
            pickle.dump(coarsed_graphs, f)

    return coarsed_graphs

def sample_expand_graph(expanded_model, devices, expand_loader,
                        enable_progress_bar=True, 
                        save_graphs=True):
    expanded_trainer = pl.Trainer(
                      strategy="ddp", 
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=devices,
                      enable_progress_bar=enable_progress_bar)
    

    start = time.time()
    expanded_graphs = []
    _expanded_graphs = expanded_trainer.predict(expanded_model, dataloaders=expand_loader)
    for graphs in _expanded_graphs:
        expanded_graphs.extend(graphs)
    

    print(f'Sampling Refine Graphs took {time.time() - start:.2f} seconds\n')

    if save_graphs:
        with open('./expanded_graphs.pkl', 'wb') as f:
            pickle.dump(expanded_graphs, f)
    return expanded_graphs


def main(global_cfg: DictConfig, coarsed_cfg: DictConfig, expanded_cfg: DictConfig, coarsed_resume: str, expanded_resume: str):
    # load coarsed model
    sample_type = global_cfg.sample_type
    if sample_type in ['all', 'coarse','from_coarse']:
        coarsed_model, coarsed_dataset_infos = load_coarsed_model(global_cfg, coarsed_cfg, coarsed_resume)

    # load coarsed model
    if sample_type in ['all','expand','from_coarse']:
        expanded_model, datamodule, expanded_dataset_infos, model_kwargs = load_expanded_model(global_cfg, expanded_cfg, expanded_resume)
        if sample_type == 'e_guidance':
            guidance_model = MaskedRegressorDiscrete.load_from_checkpoint(global_cfg.guidance_resume, **model_kwargs)
            expanded_model.guidance_model = guidance_model

    # check scaffolds
    coarsed_scaffold_data = None
    expanded_scaffold_data = None
    scaffold = global_cfg.scaffold
    if scaffold is not None:
        from src.datasets.dataset_utils import graph_from_scaffold

        coarsed_scaffold_data, expanded_scaffold_data = graph_from_scaffold(scaffold, 
                                                                            coarsed_dataset_infos,
                                                                            expanded_dataset_infos,
                                                                            )
        coarsed_graphs = [(coarsed_scaffold_data.x, 
                           coarsed_scaffold_data.x_aug, 
                           torch.zeros([1,1], dtype=torch.long, device=coarsed_scaffold_data.x.device) if \
                           coarsed_scaffold_data.edge_index.numel() < 2 else \
                           to_dense_adj(
                                edge_index=coarsed_scaffold_data.edge_index.long(),
                                edge_attr=coarsed_scaffold_data.edge_attr.long())[0]
                            )
                        ]

    # coarsed graph sample
    if sample_type in ['all', 'coarse']:
        coarsed_loader = coarsed_sample_loader(
                                    global_cfg.num_batch, 
                                    global_cfg.num_nodes, 
                                    coarsed_scaffold_data)

        coarsed_graphs = sample_coarse_graph(coarsed_model, global_cfg.devices, coarsed_loader,
                            enable_progress_bar=global_cfg.enable_progress_bar, 
                            save_graphs=global_cfg.save_graphs)

    # expanded graph sample
    if sample_type in ['all','expand','from_coarse']:
        coarsed_path = global_cfg.coarsed_path
        if coarsed_path is not None:
            with open(coarsed_path, 'rb') as f:
                coarsed_graphs = pickle.load(f)

        if sample_type in ['expand'] and coarsed_path is None:
            expand_loader = datamodule.val_dataloader()
            expand_loader = itertools.islice(expand_loader, global_cfg.num_batch)
        else:        
            expand_loader = expanded_sample_loader(
                                            global_cfg.num_batch, 
                                            global_cfg.batch_size, 
                                            coarsed_graphs,
                                            sample_per_graph=global_cfg.sample_per_graph,
                                            expanded_scaffold=expanded_scaffold_data,
                                            mask_scaffold=global_cfg.mask_scaffold
                                            )
        expanded_graphs = sample_expand_graph(expanded_model, global_cfg.devices, expand_loader,
                            enable_progress_bar=global_cfg.enable_progress_bar, 
                            save_graphs=global_cfg.save_graphs)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', type=int, default=128)
    parser.add_argument('-nb','--num_batch', type=int,  default=8)
    parser.add_argument('-k','--keep_chain', type=int)
    parser.add_argument('-c','--number_chain_steps', type=int)
    parser.add_argument('-s','--save_final', type=int)
    parser.add_argument('-r', '--resample_step', type=int, default=1)
    parser.add_argument('-n', '--num_nodes', type=int, default=None)

    parser.add_argument('--scaffold', type=str, default=None)
    parser.add_argument('--mask_scaffold', type=bool, default=False)
    parser.add_argument('--sample_per_graph', type=int, default=1, help='sample n graph for every coarsed graph')

    parser.add_argument('--expanded_cfg', type=str)
    parser.add_argument('--coarsed_cfg', type=str)
    parser.add_argument('--expanded_resume', type=str)
    parser.add_argument('--coarsed_resume', type=str)
    parser.add_argument('--guidance_resume', type=str)
    
    parser.add_argument('--skip_t',type=int)
    parser.add_argument('--target_nodes',type=int)
    parser.add_argument('--extra_nodes',type=int)

    parser.add_argument('--devices', nargs='+', type=int)
    parser.add_argument('--save_graphs', type=bool, default=False)
    parser.add_argument('--enable_progress_bar', type=bool, default=True)
    parser.add_argument('--sample_type', type=str, default='all', help='["all", "coarse", "expand", "from_coarse"]')
    parser.add_argument("--coarsed_path", type=str)
    
    args = parser.parse_args()

    global_cfg = DictConfig(vars(args))

    coarsed_cfg = None
    coarsed_resume = None
    expanded_cfg = None
    expanded_resume = None
    if args.coarsed_cfg is not None:
        with open(args.coarsed_cfg, 'r') as file:
            coarsed_cfg = yaml.safe_load(file)
        coarsed_cfg = DictConfig(coarsed_cfg)
        coarsed_cfg.train.batch_size = args.batch_size
        coarsed_resume = args.coarsed_resume
    if args.expanded_cfg is not None:
        with open(args.expanded_cfg, 'r') as file:
            expanded_cfg = yaml.safe_load(file)
        expanded_cfg = DictConfig(expanded_cfg)
        expanded_cfg.train.batch_size = args.batch_size
        expanded_resume = args.expanded_resume



    main(global_cfg, coarsed_cfg, expanded_cfg, coarsed_resume, expanded_resume)
