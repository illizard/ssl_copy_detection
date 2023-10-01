# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# MODI 230801
from collections import OrderedDict

import logging
import os
import torch
import pytorch_lightning as pl
from classy_vision.generic.distributed_util import get_rank, get_world_size, barrier
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.plugins import DDPSpawnPlugin

from torch.utils.data import DataLoader
from sscd.train import SSCD
from sscd.models.model import Model
from sscd.lib.util import call_using_args, parse_bool


logger = logging.getLogger("inference.py")
logger.setLevel(logging.INFO)

#여기서는 배치별로 처리하는 거지, 배치끼리 합치는 것은 없음, 그거는 이 나중인듯
class InferenceModel(pl.LightningModule):
    """Wraps a model for inference."""

    def __init__(self, model, metadata_keys):
        super().__init__()
        self.model = model
        self.metadata_keys = metadata_keys

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        input = batch["input"]
        batch = {k: v for (k, v) in batch.items() if k in self.metadata_keys}
        batch["embeddings"] = self(input)

        # Workaround for a CUDA synchronization bug in PyTorch Lightning.
        # Fixed upstream:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11287
        batch = {k: v.cpu() for (k, v) in batch.items()}
        # batch {'instance_id': tensor([1, 3, 5, 7],...='cuda:1'), 'split': tensor([0, 0, 0, 0],...='cuda:1'), 'image_num': tensor([277154, 4506...='cuda:1'), 'embeddings': tensor([[ 0.0459, -0...='cuda:1')}
        return batch


class Inference:
    @classmethod
    def add_parser_args(cls, parser):
        parser.add_argument("--checkpoint")
        parser.add_argument("--features")
        parser.add_argument("--model_state")
        parser.add_argument("--output_path", required=True)
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--accelerator", default="auto")
        parser.add_argument("--nodes", default=1, type=int)
        parser.add_argument("--workers", default=10, type=int)
        parser.add_argument(
            "--size", default=288, type=int, help="Image size for inference"
        )
        parser.add_argument("--preserve_aspect_ratio", default=False, type=parse_bool)
        # These options are only used if --model_state is provided.
        Model.add_arguments(parser)

    @classmethod
    def inference(cls, args, dataset, base_name="predictions"):
        if args.features:
            logger.info("Loading features")
            if os.path.exists(args.features):
                features_fn = args.features
            else:
                features_fn = f"{args.features}/{base_name}.pt"
            outputs = torch.load(features_fn, map_location=torch.device("cpu"))
        elif args.checkpoint or args.model_state:
            logger.info("Loading model")
            if args.checkpoint:
                pl_model = SSCD.load_from_checkpoint(
                    args.checkpoint, map_location=torch.device("cpu")
                )
            else:
                model = call_using_args(Model, args)
                state = torch.load(args.model_state, map_location=torch.device("cpu")) # --model_state=./ckpt/lightning_logs/version_127/checkpoints/epoch\=49-step\=19499.ckpt
                # model.load_state_dict(state, strict=True)

                # # Assuming 'model' is your PyTorch model
                # print(f"===="*30)
                # print(f"named_parameters")
                # for name, param in model.named_parameters():
                #     print(name)

                print(f"===="*30)
                print(f"named_modules")
                # Or, to print the names of the modules (layers)
                for name, module in model.named_modules():
                    print(name)
                # print(model)
                print(f"===="*30)
                print(f"current state is")
                new_state_dict = OrderedDict()
                
                for k, v in state['state_dict'].items():
                    new_key = k.replace("model.", "")
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict, strict=True)


#                for k, v in state['state_dict'].items():
#                    new_key = k.replace("model.", "")
#                    new_key = new_key.replace("backbone.vit_backbone.", "backbone.vit_backbone.model.")
#                    new_state_dict[new_key] = v
#                model.load_state_dict(new_state_dict, strict=True)

                pl_model = InferenceModel(model, ["image_num", "split", "instance_id"])
            logger.info("Creating dataloader")
            dataloader = DataLoader(
                dataset,
                # batch_size=1 if args.preserve_aspect_ratio else 256,
                # batch_size=2048,
                batch_size=128,
                num_workers=args.workers,
                persistent_workers=(
                    args.workers > 0
                ),  # unnecessary here, but silences warning
            )
            writer = InferenceWriter(args.output_path, base_name)
            trainer = pl.Trainer(
                devices=args.gpus,
                num_nodes=args.nodes,
                accelerator=args.accelerator,
                default_root_dir=args.output_path,
                strategy=DDPSpawnPlugin(find_unused_parameters=False),
                callbacks=[writer],
                log_every_n_steps=1,
            )
            logger.info("Starting inference")
            trainer.predict(pl_model, dataloaders=dataloader)
            logger.info("Loading features")
            outputs = writer.read()
        else:
            raise ValueError("Either --checkpoint or --features is required")

        logger.info("Deduplication")
        outputs = SSCD.dedup_outputs(outputs)
        return outputs

# 여기서 모든 배치에서 나온 값들 다 종합
def coalesce_outputs(outputs):
    keys = outputs[0].keys()
    return {k: torch.cat([out[k] for out in outputs]) for k in keys}


class InferenceWriter(BasePredictionWriter):
    def __init__(self, output_path: str, filename: str):
        super().__init__("epoch")
        self.output_path = output_path
        self.filename = filename
        self.output_file = os.path.join(self.output_path, f"{filename}.pt")

    def _rank_fn(self, i):
        return os.path.join(self.output_path, f"{self.filename}_rank_{i}.pt")

    def write_on_epoch_end(self, trainer, module, predictions, batch_indices):
        rank = get_rank()
        assert len(predictions) == 1
        predictions = predictions[0]
        outputs = coalesce_outputs(predictions)
        logger.info(
            "Writing %d outputs for worker %d", outputs["embeddings"].size(0), rank
        )
        torch.save(outputs, self._rank_fn(rank))
        del outputs
        logger.info("Rank %d done. Waiting for peers.", rank)
        barrier()
        if rank == 0:
            logger.info("Combining prediction outputs.")
            worker_output_fns = [self._rank_fn(i) for i in range(get_world_size())]
            worker_outputs = [torch.load(fn) for fn in worker_output_fns]
            outputs = coalesce_outputs(worker_outputs)
            del worker_outputs
            torch.save(outputs, self.output_file)
            logger.info("Save completed.")
            for fn in worker_output_fns:
                os.remove(fn)

    def read(self):
        return torch.load(self.output_file)
