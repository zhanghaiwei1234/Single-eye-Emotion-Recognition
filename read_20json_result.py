import json
import os

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

def acc_acc(test_20_results):
    acc_emotion = {'angry': '0', 'happiness': '0', 'sadness': '0', 'neutral': '0', 'fear': '0', 'surprise': '0',
                   'disgust': '0'}
    uar_all = 0
    war_all = 0
    acc_situation = {'normal': '0', 'Overexposed': '0', 'Low': '0', 'HDR': '0'}

    for i in range(20):
        inference_results = test_20_results[i]
        emotions_all_test, acc_emotion_num, situation_all_test, squence = read_emotion(inference_results)
        acc_emotion_ones = write_acc_emotion(emotions_all_test, acc_emotion_num)

        uar = UAR(acc_emotion_ones)
        war = WAR(emotions_all_test, acc_emotion_num)
        acc_situation_ones = situation(situation_all_test, squence)
        uar_all += uar
        war_all += war

        for k,v in acc_emotion_ones.items():
            acc_emotion[k] = float(v) + float(acc_emotion[k])
            #print(k,acc_emotion[k]/20)

        for k,v in acc_situation_ones.items():
            acc_situation[k] = float(v) + float(acc_situation[k])

    print('&' ,'{:.1f}'.format((float(acc_emotion["happiness"])/20)*100), '&',
           '{:.1f}'.format((float(acc_emotion["sadness"])/20)*100), '&',
           '{:.1f}'.format((float(acc_emotion["angry"])/20)*100), '&',
          '{:.1f}'.format((float(acc_emotion["disgust"]) / 20) * 100), '&',
           '{:.1f}'.format((float(acc_emotion["surprise"])/20)*100), '&',
           '{:.1f}'.format((float(acc_emotion["fear"])/20)*100), '&',
           '{:.1f}'.format((float(acc_emotion["neutral"])/20)*100), '&',

          '{:.1f}'.format((float(acc_situation["normal"]) / 20) * 100), '&' ,
            '{:.1f}'.format((float(acc_situation["Overexposed"])/20)*100), '&' ,
          '{:.1f}'.format((float(acc_situation["Low"]) / 20) * 100), '&' ,
             '{:.1f}'.format((float(acc_situation["HDR"])/20)*100), '&',

           '{:.1f}'.format((war_all / 20) * 100), '&',
            '{:.1f}'.format((uar_all / 20) * 100), '&')

if __name__ == '__main__':

    root_json = 'C:\\Users\\zhw\\Desktop\\modify-seen\\ours_results'
    test_20_results = []
    for type in os.listdir(root_json):
        alllll = []
        print('-------------------------------------')
        print(type)
        print('-------------------------------------')
        test_20_results = []
        for test_num in os.listdir(os.path.join(root_json, type)):
            if test_num[-4:] == 'json':
                lu = os.path.join(root_json, type, test_num)
                data = json.load(open(lu))
                test_20_results.append(data)
        acc_acc(test_20_results)
     





            






