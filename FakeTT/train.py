from typing import Dict
import torch
from tqdm import tqdm
'''
here, 在执行完”collate_fn=SVFEND_collate_fn“会直接跳转到这里--->for batch in tqdm(train_loader):

'''
def train(modelConfig: Dict, train_loader, trainer, criterion, optimizer):

    results = []
    truths = []
    trainer.train()
    total_loss = 0.0
    total_batch_size = 0

    for batch in tqdm(train_loader):
        batch_size = batch["label"].size(0)
        texts = batch["text"]
        audios = batch["audioframes"]
        videos = batch["frames"]
        comments = batch["comments"]
        labels = batch["label"]
        c3d = batch["c3d"]
        user_intro = batch["user_intro"]
        gpt_description = batch["gpt_description"]
        vid = batch["vid"]
        total_batch_size += batch_size
        if torch.cuda.is_available():
            audios = audios.cuda()
            texts = texts.cuda()
            videos = videos.cuda()
            comments = comments.cuda()
            labels = labels.cuda()
            labels = labels.squeeze()
            c3d = c3d.cuda()
            user_intro = user_intro.cuda()
            gpt_description = gpt_description.cuda()

        loss, pred, pred_= trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description)
        _, y = torch.max(pred, 1)
        diffusion_loss = loss.sum() #/ 1000.
        bce_loss_output = criterion(pred, labels)  #尼玛，整型和浮点型折腾来折腾去，不就是个破数嘛，我顶你个肺
        bce_loss_output_ = criterion(pred_, labels)

        results.append(y)
        truths.append(labels)

        loss_output = (torch.abs(bce_loss_output - 0.1) + 0.1) + (torch.abs(bce_loss_output_ - 0.1) + 0.1)*0.1 + diffusion_loss*0.5
        total_loss += loss_output.item()
        optimizer.zero_grad()
        loss_output.backward(loss_output)
        optimizer.step()

    results = torch.cat(results)
    truths = torch.cat(truths)
    return total_loss, results, truths


def valid(loader, trainer, criterion, modelConfig: Dict):
    trainer.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    pred_json_list = []
    pred_score = []
    pred_score_s2 = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_size = batch["label"].size(0)
            texts = batch["text"]
            audios = batch["audioframes"]
            videos = batch["frames"]
            comments = batch["comments"]
            labels = batch["label"]
            c3d = batch["c3d"]
            user_intro = batch["user_intro"]
            gpt_description = batch["gpt_description"]
            vid = batch["vid"]
            total_batch_size += batch_size
            if torch.cuda.is_available():
                audios = audios.cuda()
                texts = texts.cuda()
                videos = videos.cuda()
                comments = comments.cuda()
                labels = labels.cuda()
                labels = labels.squeeze()
                c3d = c3d.cuda()
                user_intro = user_intro.cuda()
                gpt_description = gpt_description.cuda()

            loss, pred, pred_ = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description)
            # here is the [0,1] score
            confidence_score = torch.softmax(pred, dim=1)


            #
            _, y = torch.max(pred, 1)
            diffusion_loss = loss.sum() #/ 1000.
            bce_loss_output = criterion(pred, labels)   #这里也要改成相应的格式，同上
            bce_loss_output_ = criterion(pred_, labels)

            # 记录 vid 对应预测结果
            for v, p, l, s, s2, c in zip(vid, y.cpu().tolist(), labels.cpu().tolist(), pred.cpu().tolist(), pred_.cpu().tolist(), confidence_score.cpu().tolist()):
                record = {
                    "video_id": v,
                    "predict_result": "real" if p == 0 else "fake",
                    "annotation": "real" if l == 0 else "fake",
                    "real_score": f"{s[0]:.4f}",
                    "fake_score": f"{s[1]:.4f}",
                    "real_score_s2": f"{s2[0]:.4f}",
                    "fake_score_s2": f"{s2[1]:.4f}",
                    "confidence_score_REAL": f"{c[0]:.4f}",
                    "confidence_score_FAKE": f"{c[1]:.4f}",
                    "confidence_score_delta": f"{abs(c[1]-c[0]):.4f}"
                }
                pred_json_list.append(record)
                pred_record = [s[0], s[1], l]
                pred_score.append(pred_record)

                pred_record_s2 = [s2[0], s2[1], l]
                pred_score_s2.append(pred_record_s2)

            results.append(y)
            truths.append(labels)

            loss_output = (torch.abs(bce_loss_output - 0.1) + 0.1) + (torch.abs(bce_loss_output_ - 0.1) + 0.1)*0.1 + diffusion_loss * 0.5
            total_loss += loss_output.item()

        results = torch.cat(results)
        truths = torch.cat(truths)
    return total_loss, results, truths, pred_json_list, pred_score, pred_score_s2