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
        overall_loss, untouched_loss, seen_loss, seen_frac, new_loss, new_frac = self.get_loss(
            cand_ids, output, labels
        )

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
            "untouched_loss": untouched_loss,
            "seen_loss": seen_loss,
            "seen_frac": seen_frac,
            "new_loss": new_loss,
            "new_frac": new_frac,
            "logits": output["logits"],
            "cls_labels": cls_labels,
            "cls_preds": cls_preds
        }
        
        return outputs


    def get_loss(self, cand_input_ids, output, labels):
        if self.args.manual_loss:
            overall_loss = torch.tensor(0.0).to(self.model.device)
            untouched_loss = torch.tensor(0.0).to(self.model.device)
            seen_loss = torch.tensor(0.0).to(self.model.device)
            new_loss = torch.tensor(0.0).to(self.model.device)
            seen_count = 0
            new_count = 0
            prob_outputs = torch.softmax(output["logits"], dim = 2)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    loss_i_j = -torch.log(prob_outputs[i, j, labels[i, j]])
                    untouched_loss += loss_i_j
                    # seen vs new tokens 
                    if not(labels[i, j] in cand_input_ids[i]):
                        new_count += 1
                        new_loss += loss_i_j
                        if self.args.weight_new_tokens:
                            loss_i_j *= self.args.new_tokens_weight
                    else:
                        seen_count += 1
                        seen_loss += loss_i_j
                    # long sequences 
                    if self.args.weight_long_sequences:
                        weight = 1 + self.args.long_sequences_weight * ((j + 1) / labels.shape[1])
                        loss_i_j *= weight
                    overall_loss += loss_i_j
            overall_loss /= (labels.shape[0] * labels.shape[1])
            untouched_loss /= (labels.shape[0] * labels.shape[1])
            if seen_count > 0:
                seen_loss /= seen_count
            seen_frac = 100 * seen_count / (labels.shape[0] * labels.shape[1])
            if new_count > 0:
                new_loss /= new_count
            new_frac = 100 * new_count / (labels.shape[0] * labels.shape[1])
        else:
            overall_loss = output["loss"]
            untouched_loss = output["loss"]
            seen_loss = output["loss"]
            new_loss = output["loss"]
            seen_frac = 100
            new_frac = 100
        overall_loss = torch.nan_to_num(overall_loss)
        seen_loss = torch.nan_to_num(seen_loss)
        new_loss = torch.nan_to_num(new_loss)

        return overall_loss, untouched_loss, seen_loss, seen_frac, new_loss, new_frac

