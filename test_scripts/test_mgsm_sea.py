import os
from dataclasses import field, dataclass
from typing import Optional, Any
import sys

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import random
from itertools import groupby
import pdb
import re
import multiprocessing
import threading
import json
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


from datasets import load_dataset


random.seed(112)

import torch


# model_name ="/mnt/workspace/workgroup/workgroup_v100/yiran/Babel/Model/Qwen1.5/Sea_Th_CONT_TM_ALL_ALL/Qwen2_Th"
# model_name ="/mnt/workspace/workgroup/workgroup_v100/yiran/Qwen2-1.5b"

# token_address = "/mnt/workspace/workgroup/workgroup_v100/huggingface/meta-llama/Meta-Llama-3-8B/"

# token_address = "/mnt/workspace/workgroup/workgroup_v100/yiran/Llama2-13b-base"

# token_address = "/mnt/workspace/workgroup/workgroup_v100/yiran/Mistral-nemo-base/"

# tokenizer = AutoTokenizer.from_pretrained(token_address)



def Prompting(tokenizer, model, instruction, question, detect):

    prompt = instruction + question

    #print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':100, 'do_sample':True})
    answer = tokenizer.decode(outputs[0]).replace('</s>', '')

    matches = [m.start() for m in re.finditer(detect, answer)]

    # Get the substring between the fifth and sixth occurrences
    if len(matches) >= 7:
        answer = answer[matches[5]:matches[6]]
    else:
        answer = answer

    print(answer)

    try:
        answer = re.findall(r'####\s(.+)', answer)[0]
        prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
        prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
        prd = int(prd)
        answer = int(prd)

    except:
        try:
            prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
            prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
            answer = int(prd)
        except:
            answer = -1
    
    return answer

def extract_number(answer):
    answer = re.findall(r'####\s(.+)', answer)[0]
    prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
    prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
    prd = int(prd)
    answer = int(prd)

    return answer



def main(model_name, lang):

    # model_name = "/mnt/workspace/workgroup/workgroup_v100/yiran/pruning_gemma_merge_lsn_ffn_"+argv[1]+"_"+argv[2]+"_4096_14336_40/"
    # model_name = "/mnt/workspace/workgroup/workgroup_v100/yiran/Mistral-nemo-base/"
    # model_name = "/mnt/workspace/workgroup/workgroup_v100/huggingface/meta-llama/Meta-Llama-3-8B/"
    # model_name = "/mnt/workspace/workgroup/workgroup_v100/huggingface/meta-llama/Meta-Llama-3-70B/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    full_name = {'en':'english', 'zh':'chinese', 'th':'thai', 'de':'german', 'fr':'french', 'es':'spanish', 'ru':'russian', 'vi':'vietnamese', 'id':'indonesian'}

    # dataset = list(load_dataset("/mnt/workspace/yiran/dataset/mgsm", argv[0])["test"])

    dataset = list(load_dataset("openai/gsm8k", 'main')["test"])


    task_instruction_set = {
        'english':'Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\nAnswer: Let\'s think step by step.\nRoger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.\n\nQuestion: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: Let\'s think step by step.\nThere are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Let\'s think step by step.\nLeah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQuestion: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: Let\'s think step by step.\nHe has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQuestion: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Let\'s think step by step.\nMichael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33.\n\n', 
        'chinese':"问题：罗杰有 5 个网球。 他又买了 2 罐网球。 每罐有 3 个网球。 他现在有多少个网球？\n答案：请逐步解答。罗杰一开始有 5 个球。 2 罐 3 个网球，每罐有 6 个网球。 5 + 6 = 11。答案是11。\n\n问题：服务器机房有9台计算机。 从周一到周四，每天都会安装五台计算机。 现在服务器机房有多少台电脑？\n答案：请逐步解答。周一到周四一共有4天。 每天添加 5 台计算机。 这意味着总共添加了 4 * 5 = 20 台计算机。 一开始有 9 台计算机，所以现在有 9 + 20 = 29 台计算机。 答案是 29。\n\n问题：Leah 有 32 块巧克力，她的姐姐有 42 块。如果他们吃了 35 块，总共还剩下多少块？\n答案：请逐步解答。Leah 有 32 块巧克力，Leah 的姐姐有 42 块。这意味着 最初有 32 + 42 = 74 块巧克力。 已经吃掉35个了。 所以他们总共还有 74 - 35 = 39 块巧克力。 答案是 39。\n\n问题：肖恩有五个玩具。 圣诞节时，他从爸爸妈妈那里各收到了两个玩具。 他现在有多少个玩具？\n答案：请逐步解答。他有 5 个玩具。 他从妈妈那里得到了 2 个，所以之后他就有了 5 + 2 = 7 个玩具。 然后他又从爸爸那里得到了 2 个，所以他总共有 7 + 2 = 9 个玩具。 答案是 9。\n\n问题：迈克尔有 58 个高尔夫球。 周二，他丢了 23 个高尔夫球。 周三，他又输了两场。 星期三结束时他有多少个高尔夫球？\n答案：请逐步解答。迈克尔一开始有 58 个高尔夫球，后来丢了 23 个，所以他有 58 - 23 = 35 个。在他又丢了 2 个之后，他现在有 35 - 2 = 33 个球 。 答案是 33。\n\n", 
        'thai':"คำถาม: โรเจอร์มีลูกเทนนิส 5 ลูก เขาซื้อลูกเทนนิสเพิ่มอีก 2 กระป๋อง แต่ละกระป๋องมีลูกเทนนิส 3 ลูก ตอนนี้เขามีลูกเทนนิสกี่ลูก\nคำตอบ: ลองคิดทีละขั้นตอน โรเจอร์เริ่มต้นด้วย 5 ลูก ลูกเทนนิส 3 ลูก 2 กระป๋องๆ ละ 6 ลูก 5 + 6 = 11 คำตอบคือ 11\n\nคำถาม: มีคอมพิวเตอร์เก้าเครื่องในห้องเซิร์ฟเวอร์ มีการติดตั้งคอมพิวเตอร์เพิ่มอีกห้าเครื่องในแต่ละวัน ตั้งแต่วันจันทร์ถึงพฤหัสบดี ขณะนี้มีคอมพิวเตอร์กี่เครื่องในห้องเซิร์ฟเวอร์\nคำตอบ: ลองคิดทีละขั้นตอน มี 4 วันตั้งแต่วันจันทร์ถึงวันพฤหัสบดี มีการเพิ่มคอมพิวเตอร์ 5 เครื่องในแต่ละวัน นั่นหมายความว่ามีการเพิ่มคอมพิวเตอร์ทั้งหมด 4 * 5 = 20 เครื่อง ในตอนแรกมีคอมพิวเตอร์ 9 เครื่อง ตอนนี้มี 9 + 20 = 29 เครื่อง คำตอบคือ 29\n\nคำถาม: ลีอาห์มีช็อกโกแลต 32 ชิ้น และน้องสาวของเธอมี 42 ชิ้น ถ้าพวกเขากินไป 35 ชิ้น จะเหลือทั้งหมดกี่ชิ้น?\nคำตอบ: ลองคิดทีละขั้นตอน ลีอาห์มี ช็อคโกแลต 32 อัน และน้องสาวของลีอาห์มี 42 อัน นั่นหมายความว่าแต่เดิมมีช็อกโกแลต 32 + 42 = 74 อัน กินไปแล้ว 35 อัน โดยรวมแล้วยังมีช็อกโกแลตอยู่ 74 - 35 = 39 อัน คำตอบคือ 39\n\nคำถาม: ชอว์นมีของเล่นห้าชิ้น ในวันคริสต์มาส เขาได้รับของเล่นสองชิ้นจากพ่อและแม่ของเขาอย่างละสองชิ้น ตอนนี้เขามีของเล่นกี่ชิ้น?\nคำตอบ: ลองคิดทีละขั้นตอน เขามีของเล่น 5 ชิ้น เขาได้มาจากแม่ 2 ชิ้น หลังจากนั้นเขาก็มีของเล่น 5 + 2 = 7 ชิ้น จากนั้นเขาได้มาจากพ่ออีก 2 ชิ้น รวมแล้วเขามีของเล่น 7 + 2 = 9 ชิ้น คำตอบคือ 9\n\nคำถาม: ไมเคิลมีลูกกอล์ฟ 58 ลูก ในวันอังคาร เขาสูญเสียลูกกอล์ฟไป 23 ลูก วันพุธเขาแพ้อีก 2 ปลายวันพุธเขามีลูกกอล์ฟกี่ลูก\nคำตอบ: ลองคิดทีละขั้นตอน ไมเคิลเริ่มต้นด้วยลูกกอล์ฟ 58 ลูกและเสียไป 23 ลูก ดังนั้นเขาจึงมี 58 - 23 = 35 หลังจากที่เขาเสียไปอีก 2 ลูก ตอนนี้เขามี 35 - 2 = 33 ลูกแล้ว คำตอบคือ 33\n\n",
        'vietnamese':"Câu hỏi: Roger có 5 quả bóng tennis. Anh ấy mua thêm 2 lon bóng tennis. Mỗi lon có 3 quả bóng tennis. Bây giờ anh ấy có bao nhiêu quả bóng tennis?\nTrả lời: Hãy suy nghĩ từng bước một. Roger bắt đầu với 5 quả bóng. 2 lon 3 quả bóng tennis mỗi hộp có 6 quả bóng tennis. 5 + 6 = 11. Câu trả lời là 11.\n\nCâu hỏi: Có chín máy tính trong phòng máy chủ. Năm máy tính nữa được lắp đặt mỗi ngày, từ thứ Hai đến thứ Năm. Hiện nay có bao nhiêu máy tính trong phòng máy chủ?\nTrả lời: Hãy suy nghĩ từng bước một. Có 4 ngày từ thứ hai đến thứ năm. 5 máy tính được thêm vào mỗi ngày. Điều đó có nghĩa là tổng số 4 * 5 = 20 máy tính đã được thêm vào. Lúc đầu có 9 máy tính nên hiện nay có 9 + 20 = 29 máy tính. Câu trả lời là 29.\n\nCâu hỏi: Leah có 32 viên sôcôla và em gái cô ấy có 42 viên. Nếu họ ăn 35 viên thì tổng cộng họ còn lại bao nhiêu viên?\nTrả lời: Hãy suy nghĩ từng bước một. Leah có 32 viên sôcôla và chị gái của Leah có 42 viên. Điều đó có nghĩa là ban đầu có 32 + 42 = 74 viên sôcôla. 35 đã được ăn. Vậy tổng cộng họ vẫn còn 74 - 35 = 39 sôcôla. Câu trả lời là 39.\n\nCâu hỏi: Shawn có năm món đồ chơi. Vào dịp Giáng sinh, anh ấy nhận được hai món đồ chơi từ bố và mẹ. Bây giờ bé có bao nhiêu đồ chơi?\nTrả lời: Hãy suy nghĩ từng bước một. Anh ấy có 5 món đồ chơi. Bé được mẹ tặng 2 nên sau đó bé có 5 + 2 = 7 đồ chơi. Sau đó bé được bố tặng thêm 2 món nữa nên tổng cộng bé có 7 + 2 = 9 đồ chơi. Câu trả lời là 9.\n\nCâu hỏi: Michael có 58 quả bóng gôn. Vào thứ ba, anh ấy làm mất 23 quả bóng gôn. Vào thứ Tư, anh ta mất thêm 2 quả nữa. Vào cuối ngày thứ tư, anh ấy có bao nhiêu quả bóng gôn?\nTrả lời: Hãy suy nghĩ từng bước một. Michael bắt đầu với 58 quả bóng gôn và thua 23, vậy anh ấy có 58 - 23 = 35. Sau khi mất thêm 2 quả nữa, bây giờ anh ấy có 35 - 2 = 33 quả bóng. Câu trả lời là 33.\n\n",
        'indonesian':"Pertanyaan: Roger mempunyai 5 bola tenis. Dia membeli 2 kaleng bola tenis lagi. Setiap kaleng berisi 3 bola tenis. Berapa banyak bola tenis yang dia miliki sekarang?\nJawaban: Mari kita pikirkan langkah demi langkah. Roger memulai dengan 5 bola. 2 kaleng berisi 3 bola tenis masing-masing berisi 6 bola tenis. 5 + 6 = 11. Jawabannya adalah 11.\n\nPertanyaan: Ada sembilan komputer di ruang server. Lima komputer lagi dipasang setiap hari, dari Senin hingga Kamis. Berapa banyak komputer yang ada di ruang server saat ini?\nJawab: Mari kita pikirkan langkah demi langkah. Ada 4 hari dari senin sampai kamis. 5 komputer ditambahkan setiap hari. Artinya totalnya 4 * 5 = 20 komputer ditambahkan. Awalnya ada 9 komputer, jadi sekarang ada 9 + 20 = 29 komputer. Jawabannya adalah 29.\n\nPertanyaan: Leah punya 32 coklat dan adiknya punya 42. Jika mereka makan 35, berapa jumlah totalnya yang tersisa?\nJawaban: Mari kita pikirkan langkah demi langkah. Leah punya 32 coklat dan adik Leah punya 42 coklat. Berarti awalnya ada 32 + 42 = 74 coklat. 35 telah dimakan. Jadi totalnya mereka masih punya 74 - 35 = 39 coklat. Jawabannya 39.\n\nPertanyaan: Shawn punya lima mainan. Untuk Natal, dia mendapat dua mainan masing-masing dari ibu dan ayahnya. Berapa banyak mainan yang dia miliki sekarang?\nJawaban: Mari kita pikirkan langkah demi langkah. Dia mempunyai 5 mainan. Dia mendapat 2 mainan dari ibunya, jadi setelah itu dia punya 5 + 2 = 7 mainan. Lalu dia mendapat 2 mainan lagi dari ayah, jadi totalnya dia punya 7 + 2 = 9 mainan. Jawabannya 9.\n\nPertanyaan: Michael mempunyai 58 bola golf. Pada hari Selasa, dia kehilangan 23 bola golf. Pada hari Rabu, dia kehilangan 2 lagi. Berapa banyak bola golf yang dia miliki pada hari Rabu akhir?\nJawab: Mari kita pikirkan langkah demi langkah. Michael memulai dengan 58 bola golf dan kalah 23, jadi dia mempunyai 58 - 23 = 35. Setelah dia kehilangan 2 bola lagi, sekarang dia mempunyai 35 - 2 = 33 bola. Jawabannya adalah 33.\n\n",
        'malay':'Soalan: Roger mempunyai 5 bola tenis. Dia membeli 2 tin lagi bola tenis. Setiap tin mempunyai 3 bola tenis. Berapakah bilangan bola tenis yang dia ada sekarang?\nJawapan: Mari kita fikirkan langkah demi langkah. Roger bermula dengan 5 bola. 2 tin 3 bola tenis setiap satu ialah 6 bola tenis. 5 + 6 = 11. Jawapannya ialah 11.\n\nSoalan: Terdapat sembilan komputer di dalam bilik pelayan. Lima lagi komputer dipasang setiap hari, dari Isnin hingga Khamis. Berapakah bilangan komputer sekarang dalam bilik pelayan?\nJawapan: Mari kita fikirkan langkah demi langkah. Ada 4 hari dari isnin hingga khamis. 5 komputer telah ditambah setiap hari. Ini bermakna secara keseluruhan 4 * 5 = 20 komputer telah ditambah. Terdapat 9 komputer pada mulanya, jadi sekarang terdapat 9 + 20 = 29 komputer. Jawapannya ialah 29.\n\nSoalan: Leah mempunyai 32 biji coklat dan kakaknya mempunyai 42. Jika mereka makan 35 biji, berapa keping lagi yang tinggal?\nJawapan: Mari kita fikirkan langkah demi langkah. Leah mempunyai 32 coklat dan kakak Leah mempunyai 42. Ini bermakna pada asalnya terdapat 32 + 42 = 74 coklat. 35 telah dimakan. Jadi secara keseluruhan mereka masih mempunyai 74 - 35 = 39 coklat. Jawapannya ialah 39.\n\nSoalan: Shawn ada lima mainan. Untuk Krismas, dia mendapat dua mainan setiap satu daripada ibu dan ayahnya. Berapakah bilangan mainan yang dia ada sekarang?\nJawapan: Mari kita fikirkan langkah demi langkah. Dia mempunyai 5 mainan. Dia dapat 2 daripada ibu, jadi selepas itu dia mempunyai 5 + 2 = 7 mainan. Kemudian dia mendapat 2 lagi daripada ayah, jadi secara keseluruhan dia mempunyai 7 + 2 = 9 mainan. Jawapannya ialah 9.\n\nSoalan: Michael mempunyai 58 bola golf. Pada hari Selasa, dia kehilangan 23 bola golf. Pada hari rabu, dia kehilangan 2 lagi. Berapakah bilangan bola golf yang dia miliki pada penghujung hari Rabu?\nJawapan: Mari kita fikirkan langkah demi langkah. Michael bermula dengan 58 bola golf dan kehilangan 23, jadi dia mempunyai 58 - 23 = 35. Selepas dia kehilangan 2 lagi, dia mempunyai 35 - 2 = 33 bola sekarang. Jawapannya ialah 33.\n\n',
        'german': "Frage: Roger hat 5 Tennisbälle. Er kauft 2 weitere Dosen mit Tennisbällen. Jede Dose enthält 3 Tennisbälle. Wie viele Tennisbälle hat er jetzt?\nAntwort: Lassen Sie uns Schritt für Schritt denken.\nRoger begann mit 5 Bällen. 2 Dosen mit je 3 Tennisbällen sind 6 Tennisbälle. 5 + 6 = 11. Die Antwort ist 11.\n\nFrage: Es gab neun Computer im Serverraum. Von Montag bis Donnerstag wurden jeden Tag fünf weitere Computer installiert. Wie viele Computer sind jetzt im Serverraum?\nAntwort: Lassen Sie uns Schritt für Schritt denken.\nEs gibt 4 Tage von Montag bis Donnerstag. Jeden Tag wurden 5 Computer hinzugefügt. Das bedeutet, dass insgesamt 4 * 5 = 20 Computer hinzugefügt wurden. Es gab am Anfang 9 Computer, also gibt es jetzt 9 + 20 = 29 Computer. Die Antwort ist 29.\n\nFrage: Leah hatte 32 Pralinen und ihre Schwester 42. Wenn sie 35 gegessen haben, wie viele Stücke bleiben ihnen dann insgesamt übrig?\nAntwort: Lassen Sie uns Schritt für Schritt denken.\nLeah hatte 32 Pralinen und Leahs Schwester hatte 42. Das bedeutet, dass es ursprünglich 32 + 42 = 74 Pralinen waren. 35 wurden gegessen. Insgesamt haben sie also noch 74 - 35 = 39 Pralinen. Die Antwort ist 39.\n\nFrage: Shawn hat fünf Spielzeuge. Zu Weihnachten hat er von Mama und Papa jeweils zwei Spielzeuge bekommen. Wie viele Spielzeuge hat er jetzt?\nAntwort: Lassen Sie uns Schritt für Schritt denken.\nEr hat 5 Spielzeuge. Er hat 2 von Mama bekommen, also hat er danach 5 + 2 = 7 Spielzeuge. Dann hat er noch 2 weitere von Papa bekommen, also hat er insgesamt 7 + 2 = 9 Spielzeuge. Die Antwort ist 9.\n\nFrage: Michael hatte 58 Golfbälle. Am Dienstag verlor er 23 Golfbälle. Am Mittwoch verlor er weitere 2. Wie viele Golfbälle hatte er am Ende des Mittwochs?\nAntwort: Lassen Sie uns Schritt für Schritt vorgehen.\nMichael begann mit 58 Golfbällen und verlor 23, also hat er 58 - 23 = 35. Nachdem er weitere 2 verloren hat, hat er jetzt 35 - 2 = 33 Bälle. Die Antwort ist 33.\n\n"
        }

    prompt_set_1 = {'english':'Question: ', 'chinese':"问题：", 'thai':"คำถาม: ", 'vietnamese':'Câu hỏi: ', 'indonesian':'Pertanyaan: ', 'malay':'Soalan: ', 'german':'Frage: '}
    prompt_set_2 = {'english':'\nAnswer: Let\'s think step by step.\n', 'chinese':'\n答案：请逐步解答。', 'thai':'\nคำตอบ: ลองคิดทีละขั้นตอน ', 'vietnamese':'\nTrả lời: Hãy suy nghĩ từng bước một. ', 'indonesian':'\nJawaban: Mari kita pikirkan langkah demi langkah. ', 'malay':'\nJawapan: Mari kita fikirkan langkah demi langkah. ', 'german':'\nAntwort: Lassen Sie uns Schritt für Schritt denken.\n'}


    correct_small = 0
    all_index = 0

    with tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            all_index += 1
            data = dataset[i]
            #print(data.keys())

            task_instruction_small = task_instruction_set[lang]
            prompt_small = prompt_set_1[lang] + data['question'] + prompt_set_2[lang]
            answer_small = Prompting(tokenizer, model, task_instruction_small, prompt_small, prompt_set_1[lang].strip(' '))

            # task_instruction_en = """Let\'s this step by step.\n"""
            # task_instruction_en =  task_instruction_set['en']

            # prompt_en = "Question: " + data['question'] + '\nAnswer: '
            # answer_en = Prompting(task_instruction_small, prompt_small)


            # if extract_number(data['answer']) == answer_small:
            #     correct_small += 1

            if data['answer'] == answer_small:
                correct_small += 1


            #print(answer_small)
            #print(data['answer'])
            
            acc_small = correct_small / all_index

            pbar.set_postfix(acc_small=f"{acc_small:.4f}")
            pbar.update(1)


        



if __name__ == "__main__":
    lang = "english"
    model_name = "./models/base/Llama-3-8B"
    main(model_name, lang)
