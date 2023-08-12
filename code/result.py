def read_emotion(inference_results):
    data = inference_results
    data = data['results']
    emotions_all_test = {'angry': '0', 'happiness': '0', 'sadness': '0', 'neutral': '0', 'fear': '0', 'surprise': '0',
                         'disgust': '0'}
    for emotion in emotions_all_test.keys():
        for k, v in data.items():
            if emotion == k.split('_')[6]:
                emotions_all_test[emotion] = int(emotions_all_test[emotion]) + 1

    acc_emotion_num = {'angry': '0', 'happiness': '0', 'sadness': '0', 'neutral': '0', 'fear': '0', 'surprise': '0',
                       'disgust': '0'}
    for emotion in acc_emotion_num.keys():
        for k, v in data.items():
            if emotion == v[0]['label'] == k.split('_')[6]:
                acc_emotion_num[emotion] = int(acc_emotion_num[emotion]) + 1

    situation_all_test = {'normal': '0', 'Overexposed': '0', 'Low': '0', 'HDR': '0'}
    for situation in situation_all_test.keys():
        for k, v in data.items():
            if situation == k.split('_')[5]:
                situation_all_test[situation] = int(situation_all_test[situation]) + 1

    squence = []
    for k, v in data.items():
        if v[0]['label'] == k.split('_')[6]:
            squence.append(k)
    return emotions_all_test, acc_emotion_num, situation_all_test, squence


def situation(situation_all_test, squence):
    situation_acc_test = {'normal': '0', 'Overexposed': '0', 'Low': '0', 'HDR': '0'}
    for squ in squence:
        situation_acc_test[squ.split('_')[5]] = int(situation_acc_test[squ.split('_')[5]]) + 1

    acc_situation = {'normal': '0', 'Overexposed': '0', 'Low': '0', 'HDR': '0'}

    for situation in acc_situation.keys():
        if situation_acc_test[situation] != '0':
            acc_situation[situation] = float(situation_acc_test[situation]) / float(situation_all_test[situation])
        else:
            acc_situation[situation] = 0

    return acc_situation


def write_acc_emotion(emotions_all_test, acc_emotion_num):
    acc_emotion = {'angry': '0', 'happiness': '0', 'sadness': '0', 'neutral': '0', 'fear': '0', 'surprise': '0',
                   'disgust': '0'}
    for emotion in acc_emotion.keys():
        if acc_emotion_num[emotion] != '0':
            acc_emotion[emotion] = float(acc_emotion_num[emotion]) / float(emotions_all_test[emotion])
        else:
            acc_emotion[emotion] = 0

    return acc_emotion


def UAR(acc_emotion):
    all = 0
    for emotion in acc_emotion.keys():
        all += float(acc_emotion[emotion])
    uar = all / 7
    return uar


def WAR(emotions_all_test, acc_emotion_num):
    all_test = 0
    acc_num = 0
    for emotion in emotions_all_test.keys():
        all_test += float(emotions_all_test[emotion])
    for emotion in acc_emotion_num.keys():
        acc_num += float(acc_emotion_num[emotion])
    war = acc_num / all_test
    return war


def F1(uar, war):
    p = 0.33
    f1 = p * uar + (1 - p) * war
    return f1

def acc_acc(test_20_results, epoch, tb_writer):
    acc_emotion = {'angry': '0', 'happiness': '0', 'sadness': '0', 'neutral': '0', 'fear': '0', 'surprise': '0',
                   'disgust': '0'}
    uar_all = 0
    war_all = 0
    f1_all = 0
    acc_situation = {'normal': '0', 'Overexposed': '0', 'Low': '0', 'HDR': '0'}
    for i in range(len(test_20_results)): 
        print(i)
        inference_results = test_20_results[i]
        emotions_all_test, acc_emotion_num, situation_all_test, squence = read_emotion(inference_results)
        acc_emotion_ones = write_acc_emotion(emotions_all_test, acc_emotion_num)

        uar = UAR(acc_emotion_ones)
        war = WAR(emotions_all_test, acc_emotion_num)
        f1 = F1(uar, war)
        acc_situation_ones = situation(situation_all_test, squence)
        uar_all += uar
        war_all += war
        f1_all += f1
        for k,v in acc_emotion_ones.items():
            acc_emotion[k] = float(v) + float(acc_emotion[k])
        for k,v in acc_situation_ones.items():
            acc_situation[k] = float(v) + float(acc_situation[k])

        if tb_writer is not None:
            tb_writer.add_scalar('test_num/'+str(i+1), f1, epoch)

    tb_writer.add_scalar('acc/angry', acc_emotion["angry"]/20, epoch)
    tb_writer.add_scalar('acc/happiness', acc_emotion["happiness"]/20, epoch)
    tb_writer.add_scalar('acc/sadness', acc_emotion["sadness"]/20, epoch)
    tb_writer.add_scalar('acc/neutral', acc_emotion["neutral"]/20, epoch)
    tb_writer.add_scalar('acc/fear', acc_emotion["fear"]/20, epoch)
    tb_writer.add_scalar('acc/surprise', acc_emotion["surprise"]/20, epoch)
    tb_writer.add_scalar('acc/disgust', acc_emotion["disgust"]/20, epoch)

    tb_writer.add_scalar('Metrics/UAR-average', uar_all/20, epoch)
    tb_writer.add_scalar('Metrics/WAR-average', war_all/20, epoch)
    tb_writer.add_scalar('Metrics/F1-p=0.33-average', f1_all/20, epoch)

    tb_writer.add_scalar('situation/normal--average', acc_situation["normal"]/20, epoch)
    tb_writer.add_scalar('situation/Overexposed--average', acc_situation["Overexposed"]/20, epoch)
    tb_writer.add_scalar('situation/Low--average', acc_situation["Low"]/20, epoch)
    tb_writer.add_scalar('situation/HDR--average', acc_situation["HDR"]/20, epoch)
            






