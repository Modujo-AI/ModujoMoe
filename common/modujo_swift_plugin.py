"""Register Modujo Qwen4-Exp and optional QSA distillation with ms-swift."""

import logging
import os

from transformers import Seq2SeqTrainingArguments

from common.qwen4_exp_dense_compat import enable_dense_qwen4_exp_attention
from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.model.register import ModelLoader
from swift.optimizers.base import OptimizerCallback
from swift.optimizers.mapping import optimizers_map
from swift.template import TemplateType

from pretrain.muon import MuonWithAuxAdamW, use_muon_for_parameter


LOGGER = logging.getLogger(__name__)


enable_dense_qwen4_exp_attention()

# ms-swift 4.5.3's MuonOptimizerCallback reads this Swift-only attribute from
# the distilled HF TrainingArguments object. ``swift pt`` does not copy it,
# causing an AttributeError before optimizer creation. A class default keeps
# the callback's intended "auto provision the Muon repo" behavior.
if not hasattr(Seq2SeqTrainingArguments, "local_repo_path"):
    Seq2SeqTrainingArguments.local_repo_path = None


class ModujoMuonOptimizerCallback(OptimizerCallback):
    """Use Muon only for valid 2D hidden weights; route everything else to AdamW."""

    def create_optimizer(self, model=None):
        model = model or self.trainer.model
        muon_params, adamw_params = [], []
        muon_names, adamw_names = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if use_muon_for_parameter(name, parameter):
                muon_params.append(parameter)
                muon_names.append(name)
            else:
                adamw_params.append(parameter)
                adamw_names.append(name)
        args = self.args
        optimizer = MuonWithAuxAdamW(
            muon_params,
            adamw_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=float(os.environ.get("MODUJO_MUON_MOMENTUM", "0.95")),
            ns_steps=int(os.environ.get("MODUJO_MUON_NS_STEPS", "5")),
            adam_betas=(args.adam_beta1, args.adam_beta2),
            adam_eps=args.adam_epsilon,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            LOGGER.warning(
                "MODUJO_OPTIMIZER_ACTIVE class=%s muon_tensors=%d muon_parameters=%d "
                "adamw_tensors=%d adamw_parameters=%d",
                type(optimizer).__name__,
                len(muon_params),
                sum(parameter.numel() for parameter in muon_params),
                len(adamw_params),
                sum(parameter.numel() for parameter in adamw_params),
            )
            LOGGER.info("Muon sample parameters: %s", muon_names[:5])
            LOGGER.info("AdamW sample parameters: %s", adamw_names[:5])
        return optimizer


optimizers_map["modujo_muon"] = ModujoMuonOptimizerCallback


def _enable_qsa_distillation_from_env() -> None:
    if os.environ.get("MODUJO_QSA_DISTILL", "0") != "1":
        return
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    from pretrain.train_transformers import QSAIndexerDistillation

    if getattr(Qwen4ExpForCausalLM, "_modujo_distill_patched", False):
        return
    original_forward = Qwen4ExpForCausalLM.forward
    distill_weight = float(os.environ.get("MODUJO_QSA_DISTILL_WEIGHT", "1.0"))
    lm_weight = float(os.environ.get("MODUJO_QSA_LM_LOSS_WEIGHT", "0.0"))

    def forward_with_indexer_distillation(self, *args, **kwargs):
        distiller = getattr(self, "_modujo_qsa_distiller", None)
        if distiller is None:
            distiller = QSAIndexerDistillation(self)
            self._modujo_qsa_distiller = distiller
        outputs = original_forward(self, *args, **kwargs)
        indexer_loss = distiller.loss()
        outputs.loss = distill_weight * indexer_loss + lm_weight * outputs.loss
        return outputs

    Qwen4ExpForCausalLM.forward = forward_with_indexer_distillation
    Qwen4ExpForCausalLM._modujo_distill_patched = True


_enable_qsa_distillation_from_env()


register_model(
    ModelMeta(
        "modujo_qwen4_exp",
        [ModelGroup([Model(hf_model_id="Alexhu1999/Modujo-9B-A1B")])],
        ModelLoader,
        # qwen3_8 is a multimodal Swift template that expects get_rope_index().
        # This checkpoint is text-only, so the compatible Qwen3 template is used.
        template=TemplateType.qwen3,
        model_arch=ModelArch.llama,
        architectures=["Qwen4ExpForCausalLM"],
        requires=["transformers>=5.16.0"],
        tags=["text-generation", "moe"],
    ),
    exist_ok=True,
)
