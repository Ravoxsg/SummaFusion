import torch
import torch.nn as nn
import numpy as np



class ModelAbstractiveFusion(nn.Module):
    def __init__(self, model, tokenizer, args):
        super(ModelAbstractiveFusion, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.cls_loss_fct = nn.BCEWithLogitsLoss(reduction = "none")
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, mode, source_ids, source_mask, cand_ids, cand_mask, labels, cls_labels):
        cand_ids = torch.cat((source_ids, cand_ids), -1)
        cand_mask = torch.cat((source_mask, cand_mask), -1)

        if self.args.fp16:
            with torch.cuda.amp.autocast():
                output, cls_outputs = self.model(
                    input_ids = cand_ids, attention_mask = cand_mask, labels = labels, output_hidden_states = False
                )
        else:
            output, cls_outputs = self.model(
                input_ids = cand_ids, attention_mask = cand_mask, labels = labels, output_hidden_states = False
            )
        overall_loss = self.get_loss(output)

        cls_loss = torch.tensor(-1).to(self.args.device)
        cls_preds = -1 * torch.ones(cls_labels.shape).to(self.args.device)
        if self.args.classify_candidates:
            cls_labels = cls_labels.to(self.args.device)
            cls_loss_raw = self.cls_loss_fct(cls_outputs, cls_labels)
            cls_loss_mask = torch.ones(cls_labels.shape).to(cls_labels.device)
            cls_loss_mask[cls_labels == -1] = 0
            cls_loss = cls_loss_raw * cls_loss_mask
            cls_loss = cls_loss.mean()
            overall_loss = overall_loss + self.args.cls_loss_weight * cls_loss
            cls_preds = self.sigmoid(cls_outputs)

        outputs = {
            "loss": overall_loss,
            "cls_loss": cls_loss,
            "logits": output["logits"],
            "cls_labels": cls_labels,
            "cls_preds": cls_preds
        }
        
        return outputs

    def get_loss(self, output):
        overall_loss = output["loss"]
        overall_loss = torch.nan_to_num(overall_loss)

        return overall_loss

