import time
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from utils import AverageMeter
from result import acc_acc

def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs, k=min(output_topk, len(class_names)))
    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })
    return video_results

global test_20_results

def inference(data_loader, model, result_path, class_names, no_average, output_topk, epoch, tb_writer, test_num):
    print('inference', '------------', test_num)
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}

    with torch.no_grad():
        for i, (event_inputs, frame_inputs, targets) in enumerate(data_loader):
            video_ids, segments = zip(*targets)
            end_time = time.time()
            outputs = model(frame_inputs, event_inputs)
            data_time.update(time.time() - end_time)
            outputs = F.softmax(outputs, dim=1).cpu()

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({'segment': segments[j], 'output': outputs[j]})
            batch_time.update(time.time() - end_time)

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    with result_path.open('w') as f:
        json.dump(inference_results, f)

    # test 20 times
    if test_num == '1':
        global test_20_results
        test_20_results = []
    test_20_results.append(inference_results)
    print('test_results_num is : ', len(test_20_results))
    if test_num == '20':
        acc_acc(test_20_results, epoch, tb_writer)
        test_20_results = []
        






