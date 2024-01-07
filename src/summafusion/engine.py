import gc
import numpy as np
import sys
import torch
sys.path.append("/data/mathieu/2nd_stage_summarization/")
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score

from generation_utils import GenerationMixin
from common.evaluation import overall_eval, get_rouge_scores



def training_loop(train_loader, val_loader, tokenizer, model, optimizer, scheduler, args):
    if args.eval_epoch_0:
        print("\nEpoch 0 validation:")
        _, _, _, val_summaries, val_labels,  = validate(val_loader, tokenizer, model, args)
        mean_r, r1, r2, rl = quick_eval(val_summaries, val_labels, args)
        print("VALIDATION mean R: {:.4f}, R-1: {:.4f}, R-2: {:.4f}, R-L: {:.4f}".format(
            mean_r, r1, r2, rl
        ))

    total_steps = 0
    val_metrics = {
        "mean_r": [],
        "r1": [],
        "r2": [],
        "rl": []
    }
    for epoch in range(1, args.n_epochs + 1):
        print("\n", "*"*50, "New epoch, epoch {} / {}".format(epoch, args.n_epochs))
        total_steps = train(total_steps, epoch, train_loader, val_loader, tokenizer, model, optimizer, scheduler, val_metrics, args)
        if args.eval_per_epoch:
            model.eval()
            _, _, _, val_summaries, val_labels  = validate(val_loader, tokenizer, model, args)
            mean_r, r1, r2, rl = quick_eval(val_summaries, val_labels, args)
            val_metrics["mean_r"].append(mean_r)
            val_metrics["r1"].append(r1)
            val_metrics["r2"].append(r2)
            val_metrics["rl"].append(rl)
            best_mean_r = max(val_metrics["mean_r"])
            best_r1 = max(val_metrics["r1"])
            best_r2 = max(val_metrics["r2"])
            best_rl = max(val_metrics["rl"])
            print("VALIDATION mean R: {:.4f} (best: {:.4f}), R-1: {:.4f} (best: {:.4f}), R-2: {:.4f} (best: {:.4f}), R-L: {:.4f} (best: {:.4f})".format(
                mean_r, best_mean_r, r1, best_r1, r2, best_r2, rl, best_rl
            ))
            for metric in val_metrics.keys():
                if args.early_stopping_metric == metric:
                    if val_metrics[metric][-1] == max(val_metrics[metric]):
                        print("!!!New best validation performance!!!")
                        if args.save_model:
                            torch.save(model.state_dict(), args.save_model_path + "pytorch_model.bin")
                            print("saved model!", args.save_model_path)
            model.train()


def train(total_steps, epoch, train_loader, val_loader, tokenizer, model, optimizer, scheduler, val_metrics, args):
    print("\nStarting training...")
    model.train()

    just_evaluated = False
    losses, temp_losses, temp_cls_losses = [], [], []
    scaler = torch.cuda.amp.GradScaler()
    for idx, batch in tqdm(enumerate(train_loader)):
        mode = batch["mode"]
        source_ids = batch["source_ids"].to(args.device)
        source_mask = batch["source_mask"].to(args.device)
        cand_ids = batch["cand_ids"].to(args.device)
        cand_mask = batch["cand_mask"].to(args.device)
        labels = batch["label_ids"].to(args.device)
        cls_labels = batch["cls_labels"].to(args.device)

        outputs = model(mode, source_ids, source_mask, cand_ids, cand_mask, labels, cls_labels)

        del source_ids
        del source_mask
        del cand_ids
        del cand_mask
        del labels
        del cls_labels
        gc.collect()

        loss = outputs["loss"] 
        cls_loss = outputs["cls_loss"]

        losses.append(loss.item())
        temp_losses.append(loss.item())
        temp_cls_losses.append(cls_loss.item())

        loss /= args.gradient_accumulation_steps
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if ((idx + 1) % args.gradient_accumulation_steps == 0) or ((idx + 1) == len(train_loader)):
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler != None:
                scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            total_steps += 1
            just_evaluated = False
            just_printed = False

        if (total_steps > 0) and (total_steps % args.print_every == 0) and (just_printed == False):
            m_loss = np.mean(losses)
            print("\nIter: {}, loss: {:.4f}".format(total_steps, m_loss))
            losses= []
            just_printed = True

        if (total_steps > 0) and (total_steps % args.eval_every == 0) and (just_evaluated == False):
            # Training performance
            mean_loss = np.mean(temp_losses)
            mean_cls_loss = np.mean(temp_cls_losses)
            print("\n\nIter: {} (epoch {} batch {})".format(
                total_steps, epoch, idx + 1
            ))
            print("TRAINING loss: {:.4f}, CLS loss: {:.4f}".format(
                mean_loss, mean_cls_loss
            ))
            temp_losses, temp_cls_losses = [], []

            # Validation performance
            _, _, _, val_summaries, val_labels  = validate(val_loader, tokenizer, model, args)
            mean_r, r1, r2, rl = quick_eval(val_summaries, val_labels, args) 
            val_metrics["mean_r"].append(mean_r)
            val_metrics["r1"].append(r1)
            val_metrics["r2"].append(r2)
            val_metrics["rl"].append(rl)
            best_mean_r = max(val_metrics["mean_r"])
            best_r1 = max(val_metrics["r1"])
            best_r2 = max(val_metrics["r2"])
            best_rl = max(val_metrics["rl"])
            print("VALIDATION mean R: {:.4f} (best: {:.4f}), R-1: {:.4f} (best: {:.4f}), R-2: {:.4f} (best: {:.4f}), R-L: {:.4f} (best: {:.4f})".format(
                mean_r, best_mean_r, r1, best_r1, r2, best_r2, rl, best_rl
            ))
            for metric in val_metrics.keys():
                if args.early_stopping_metric == metric:
                    if val_metrics[metric][-1] == max(val_metrics[metric]):
                        print("!!!New best validation performance!!!")
                        if args.save_model:
                            torch.save(model.state_dict(), args.save_model_path)
                            print("saved model!", args.save_model_path)
            model.train()
            just_evaluated = True

    return total_steps


def validate(loader, tokenizer, model, args):
    print("\nStart evaluation...")
    model.eval()

    gm = GenerationMixin

    # train
    val_losses, val_cls_losses, val_cls_labels, val_cls_preds = [], [], [], []
    val_texts, val_candidates, val_labels, val_summaries = [], [], [], []
    for idx, batch in tqdm(enumerate(loader)):
        model.zero_grad()

        val_texts += batch["source"]
        val_candidates += batch["candidates"]

        labels = batch["label"]
        val_labels += labels
        summaries = generation_step(batch, tokenizer, model, gm, args)
        val_summaries += summaries

        del labels
        del summaries
        gc.collect()

        del batch
        gc.collect()
        if idx % 100 == 0:
            print("step: {}".format(idx))

    return val_losses, val_texts, val_candidates, val_summaries, val_labels


def generation_step(batch, tokenizer, model, gm, args):
    cand_ids = batch["cand_ids"].to(args.device)
    cand_mask = batch["cand_mask"].to(args.device)
    source_ids = batch["source_ids"].to(args.device)
    source_mask = batch["source_mask"].to(args.device)
    cand_ids = torch.cat((source_ids, cand_ids), -1)
    cand_mask = torch.cat((source_mask, cand_mask), -1)

    model.model._prepare_encoder_decoder_kwargs_for_generation = gm._prepare_encoder_decoder_kwargs_for_generation
    model.model._expand_inputs_for_generation = gm._expand_inputs_for_generation
    model.model.greedy_search = gm.greedy_search
    model.model.sample = gm.sample
    model.model.beam_search = gm.beam_search
    model.model.beam_sample = gm.beam_sample
    model.model.group_beam_search = gm.group_beam_search
    summary_ids = gm.generate(
        model.model,
        cand_ids,
        num_beams = args.num_gen_beams,
        num_return_sequences = args.num_return_sequences,
        max_length = args.max_gen_summary_length,
        repetition_penalty = args.repetition_penalty,
        length_penalty = args.length_penalty,
        no_repeat_ngram_size = args.no_repeat_ngram_size,
        early_stopping = True,
        use_cache = True,
        max_source_length = args.max_source_length,
        use_source = args.use_source,
        use_candidates = args.use_candidates
    )
    generated = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return generated


def quick_eval(summaries, labels, args):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=args.stemmer)
    all_mean_rs, all_r1s, all_r2s, all_rls = [], [], [], []
    for i in range(len(labels)):
        r1, r2, rl = get_rouge_scores(summaries[i], labels[i], scorer, args)
        all_mean_rs.append((r1 + r2 + rl)/3)
        all_r1s.append(r1)
        all_r2s.append(r2)
        all_rls.append(rl)
    mean_r = 100 * np.mean(all_mean_rs)
    r1 = 100 * np.mean(all_r1s)
    r2 = 100 * np.mean(all_r2s)
    rl = 100 * np.mean(all_rls)
    
    return mean_r, r1, r2, rl


