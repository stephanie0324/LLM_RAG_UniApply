## How To Run: ####################################
# `cd test`
# `DEBUG=false python kgi_eval.py`
###################################################

## 定義測試實驗參數: ###############################

MODEL_NAME = "Meta-Llama-3-70B-Instruct-GPTQ"
MODEL_TEMPERATURE = 0.0
DATA_TYPE_NAME = "KGI_POC_v1c.rc3"
CHAT_MODE = "KGI POC"
RETRIEVER_LIMIT = 4
TRY_NUM = 3  # 每題要試幾次，每次都正確才通過該題測試。若第一次就失敗，就不會再試第二次.

####################################################

import sys
sys.path.append("..")

import unittest
import logging
import re
from itri_api import ITRI_API
from chat_web import get_chains, get_retrieval_config
from config import settings
from prompts import prompts


def init_root_logger():
    # 禁止所有子 logger 的傳播
    logging.getLogger().setLevel(logging.NOTSET)
    for logger_name in logging.root.manager.loggerDict:
        if logger_name != "root":
            logging.getLogger(logger_name).propagate = False

    # 移除 root logger 的預設 StreamHandler
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    # 為 root logger 增加 StreamHandler，以輸出至console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)

    # 修改 root logger 的輸出格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)


init_root_logger()

llm_model = settings.MODEL_CONFIG.root.get(MODEL_NAME).asInstance()
llm_model.temperature = MODEL_TEMPERATURE

itri_api = ITRI_API(**settings.ITRI_KM_CONFIG)
retrieval_config = get_retrieval_config(itri_api)
retriever = retrieval_config.get(DATA_TYPE_NAME)
retriever.limit = RETRIEVER_LIMIT

def extract_ans_from_res(chain_res):
    """ 將 "Answer: XXXXX\n\nAnswer Snippet: YYYYYY\n\nFile Name: ZZZZZZZZZZZZZ\n\nParagraph Title: AAAAAAAAAA"
    解析為：{'Answer:': 'XXXXX', 'Answer Snippet:': 'YYYYYY', 'File Name:': 'ZZZZZZZZZZZZZ', 'Paragraph Title:': 'AAAAAAAAAA'}
    """
    lines = chain_res['answer'].split('\n')
    result = {}

    ans_keys = ['Answer:', 'Explanation:', 'Answer Snippet:', 'File Name:', 'Paragraph Title:']
    
    current_key = None
    for line in lines:
        is_key_in_line = any((key in line) for key in ans_keys)
        # 若這行包含任一key，則加到字典.
        if is_key_in_line:
            for key in ans_keys:
                if key in line:
                    if key not in result:
                        result[key] = []
                    result[key].append(line.replace(key, ""))
                    current_key = key
                    break
        # 若這行不包含key，則視為接續上一個key.
        else:
            if current_key is not None:
                result[current_key].append(line)
    for key, value in result.items():
        result[key] = '\n'.join(value).strip()
    return result

chain = get_chains(chat_mode=CHAT_MODE, llm_model=llm_model, retriever=retriever, prompts=prompts)


def boc_checker(pred_ans, gt_ans):
    """bag of char 確認：若gt_ans中所有的字元都包含於pred_ans中，則測試通過，反之raise
    """
    assert set(gt_ans).issubset(set(pred_ans)), \
        '答案錯誤: Ans缺少字元:' + str(set(gt_ans) - set(pred_ans))
    return True

class TestKgi(unittest.TestCase):

    def test_a_q1(self):
        logging.info('-' * 20)
        logging.info('Q1:')
        q = "請問全面保專案得附加意外傷害一至六級傷害失能補償保險附加條款(*PAED)之規則?"

        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                gt_ans = "金好意保險/鑫好意保險/龍滿意保險/元氣御守終身保險/鑫滿意保險/PAA/IPA/VPA且該保單傷害險保額須為100萬(含)以上"
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                boc_checker(pred_ans, gt_ans)


                assert '全面保專案_投保規則_20200228.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:全面保專案_投保規則_20200228.pdf, 預測為:{pred_file_name})'
                assert '投保限制' in pred_paragraph_title, 'Paragraph Title錯誤'
            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_b_q2(self):
        logging.info('-' * 20)
        logging.info('Q2:')
        q = "我投保了全面保專案，可以每個月交付保費嗎?"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '全面保專案_投保規則_20200228.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:全面保專案_投保規則_20200228.pdf, 預測為:{pred_file_name})'
                assert '繳別' in pred_paragraph_title, 'Paragraph Title錯誤'

                assert "不可以" in pred_ans, \
                    '答案錯誤: Ans不包含"不可以"'
            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_c_q3(self):
        logging.info('-' * 20)
        logging.info('Q3:')
        q = "最近我把金好專案解約了 可以投保全面保嗎?"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']


                assert "不可以" not in pred_ans and "可以" in pred_ans, \
                    '答案錯誤: Ans不包含"可以"'
                assert '全面保專案_投保規則_20200228.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:全面保專案_投保規則_20200228.pdf, 預測為:{pred_file_name})'
                assert '投保限制' in pred_paragraph_title, 'Paragraph Title錯誤'
            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_d_q4(self):
        logging.info('-' * 20)
        logging.info('Q4:')
        q = "台北富邦銀行台幣會天天扣款嗎?"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '103年度首期轉帳件扣款作業時間表.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:103年度首期轉帳件扣款作業時間表.pdf, 預測為:{pred_file_name})'
                assert '扣款規則' in pred_paragraph_title, 'Paragraph Title錯誤'
                assert "不會" in pred_ans or re.fullmatch(".*不.*天天扣款.*", pred_ans), \
                    '答案錯誤: Ans不包含"不會"'
            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_e_q5(self):
        logging.info('-' * 20)
        logging.info('Q5:')
        q = "請問MAHUAB繳費期間可以15年嗎?"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '凱基人壽健康樂活醫療終身保險_商品說明.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:凱基人壽健康樂活醫療終身保險_商品說明.pdf, 預測為:{pred_file_name})'
                assert '投保規則' in pred_paragraph_title, 'Paragraph Title錯誤'
                assert "不可以" in pred_ans or '不能' in pred_ans or '不包括' in pred_ans, \
                    '答案錯誤: Ans不包含"不可以"'
            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_f_q6(self):
        logging.info('-' * 20)
        logging.info('Q6:')
        q = "請比較健康樂活與新樂活商品繳費期間的差異"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '凱基人壽健康樂活醫療終身保險_商品說明.pdf' in pred_file_name and \
                    '凱基人壽新樂活終身醫療健康保險_商品說明.pdf' in pred_file_name, f'File Name錯誤(預測為:{pred_file_name})'
                assert '投保規則' in pred_paragraph_title, 'Paragraph Title錯誤'

                assert '健康樂活' in pred_ans and '新樂活' in pred_ans
                assert re.fullmatch('.*健康樂活.*(10|十).*(20|二十)[^0-9一二三四五六七八九十]*新樂活.*(15|十五).*(20|二十).*(30|三十)[^0-9一二三四五六七八九十]*', pred_ans) or \
                    re.fullmatch('.*新樂活.*(15|十五).*(20|二十).*(30|三十)[^0-9一二三四五六七八九十]*健康樂活.*(10|十).*(20|二十)[^0-9一二三四五六七八九十]*', pred_ans)

            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_g_q7(self):
        logging.info('-' * 20)
        logging.info('Q7:')
        q = "被保人14歲死亡有何相關規定"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '凱基人壽健康樂活醫療終身保險_商品說明.pdf' in pred_file_name and \
                    (
                        '凱基人壽新樂活終身醫療健康保險_商品說明.pdf' in pred_file_name or \
                        '凱基人壽新樂活終身醫療健康保險_保單條款.pdf' in pred_file_name
                    ), \
                    f'File Name錯誤(預測為:{pred_file_name})'
                boc_checker(pred_paragraph_title, '所繳保險費的退還、身故保險金或喪葬費用保險金')

                assert '健康樂活' in pred_ans and '新樂活' in pred_ans
                boc_checker(pred_ans, '所繳保險費退還予要保人')

            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_h_q8(self):
        logging.info('-' * 20)
        logging.info('Q8:')
        q = "新樂活買計畫7，每次門診手術費用可獲得多少給付金？"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '凱基人壽新樂活終身醫療健康保險_保單條款.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:凱基人壽新樂活終身醫療健康保險_保單條款.pdf, 預測為:{pred_file_name})'
                assert '保險利益' in pred_paragraph_title, 'Paragraph Title錯誤'

                assert '無計畫7' in pred_ans or \
                    '無相關' in pred_ans or \
                    '不存在' in pred_ans or \
                    re.fullmatch('.*(沒有|未).*計畫7.*', pred_ans)

            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_i_q9(self):
        logging.info('-' * 20)
        logging.info('Q9:')
        q = "華南銀行台幣帳戶如果我在4/10提出媒體申請，會幾號進行帳戶扣款?"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '103年度首期轉帳件扣款作業時間表.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:103年度首期轉帳件扣款作業時間表.pdf, 預測為:{pred_file_name})'
                assert '扣款作業時間表' in pred_paragraph_title or '扣款規則' in pred_paragraph_title, 'Paragraph Title錯誤'
                assert '4/15' in pred_ans or '4月15日' in pred_ans

            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

    def test_j_q10(self):
        logging.info('-' * 20)
        logging.info('Q10:')
        q = "16歲BDDR保額最高是多少?"
        for i in range(TRY_NUM):
            try:
                res = chain.invoke({"question": q, "chat_history": []})
                logging.info(extract_ans_from_res(res))
                pred_ans = extract_ans_from_res(res)['Answer:']
                pred_file_name = extract_ans_from_res(res)['File Name:']
                pred_paragraph_title = extract_ans_from_res(res)['Paragraph Title:']

                assert '中國人壽醫卡健康一年期重大傷病健康保險附約投保規則_20210905.pdf' in pred_file_name, \
                    f'File Name錯誤(正確:中國人壽醫卡健康一年期重大傷病健康保險附約投保規則_20210905.pdf, 預測為:{pred_file_name})'
                assert '重大疾病提前給付及特定傷病應累計商品及速算表' in pred_paragraph_title or '附件' in pred_paragraph_title, 'Paragraph Title錯誤'
                assert '100萬' in pred_ans

            except Exception as e:
                err_msg = f'{e}\n第{i+1}次測試時答題錯誤，輸出為：{extract_ans_from_res(res)}'
                logging.error(err_msg)
                raise Exception(err_msg)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKgi)
    unittest.TextTestRunner().run(suite)
