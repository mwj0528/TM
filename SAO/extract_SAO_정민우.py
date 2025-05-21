import os
import pandas as pd
import json
import nltk
import multiprocessing
from datetime import date, time, datetime
from nltk.tokenize import sent_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import time
import re
import string

# nltk.download('punkt')

"""
ready!
cmd창에

cd /d G:\
cd G:\내 드라이브\STUDY_PYTHON\BACKUP\191018_stanfordparser_corenlp\stanford-corenlp-full-2018-10-05\stanford-corenlp-full-2018-10-05
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

"""
def modify_file_for_sao(download_path_1, download_path_5, start, end):
    # download_path_1: colab으로 다운받은 원래 파일 path
    # download_path_6: 저장할 파일 위치
    allow = string.ascii_letters + string.digits + ',.:()/' + ' '
    for i6 in range(start, end + 1):
        file_name_for_5 = download_path_1 + '\\' + str(i6) + 'th_data.json'
        save_file_name_for_5 = download_path_5 + '\\' + str(i6) + 'th_data.csv'
        data = json.load(open(file_name_for_5))  # dict
        data_2 = data['data']  # list of dicts
        num = []
        raw_text = []
        for i66 in range(len(data_2)):
            num.append(data_2[i66]['publication_number'] + '_title')
            num.append(data_2[i66]['publication_number'] + '_abstract')
            num.append(data_2[i66]['publication_number'] + '_claims')
            num.append(data_2[i66]['publication_number'] + '_description')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['title_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['abstract_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['claims_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['description_localized'][0]['text']))
            except:
                raw_text.append('')
        if (len(num) == len(raw_text) and len(num) % 4 == 0): print('doc ' + str(i6) + ' is well done')
        data_for_sao = pd.DataFrame({"num": num, "raw_text": raw_text})
        data_for_sao.to_csv(save_file_name_for_5, index=False, encoding='utf-8')


def modify_file_for_sao_with_background(download_path_1, download_path_5, start, end):
    # download_path_1: colab으로 다운받은 원래 파일 path
    # download_path_6: 저장할 파일 위치
    allow = string.ascii_letters + string.digits + ',.:()/' + ' '
    for i6 in range(start, end + 1):
        file_name_for_5 = download_path_1 + '\\' + str(i6) + 'th_data.json'
        save_file_name_for_5 = download_path_5 + '\\' + str(i6) + 'th_data.csv'
        data = json.load(open(file_name_for_5))  # dict
        data_2 = data['data']  # list of dicts
        num = []
        raw_text = []
        for i66 in range(len(data_2)):
            num.append(data_2[i66]['publication_number'] + '_title')
            num.append(data_2[i66]['publication_number'] + '_abstract')
            num.append(data_2[i66]['publication_number'] + '_claims')
            num.append(data_2[i66]['publication_number'] + '_description')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['title_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['abstract_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['claims_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                text_list=re.sub('[^%s]' % allow, "", data_2[i66]['description_localized'][0]['text']).split('      ')
                start_paragraph=0
                end_paragraph = 0
                for text_i,text in enumerate(text_list):
                    if start_paragraph==0 and text=='BACKGROUND':
                        start_paragraph=text_i
                        continue
                    if start_paragraph!=0 and end_paragraph==0 and len(text)<20:
                        end_paragraph=text_i-1
                only_background_text=" ".join(text_list[start_paragraph:end_paragraph])
                raw_text.append(only_background_text)
            except:
                raw_text.append('')
        if (len(num) == len(raw_text) and len(num) % 4 == 0): print('doc ' + str(i6) + ' is well done')
        data_for_sao = pd.DataFrame({"num": num, "raw_text": raw_text})
        data_for_sao.to_csv(save_file_name_for_5, index=False, encoding='utf-8')


def modify_file_for_sao_except_description(download_path_1, download_path_5, start, end):
    # download_path_1: colab으로 다운받은 원래 파일 path
    # download_path_6: 저장할 파일 위치
    allow = string.ascii_letters + string.digits + ',.:()/' + ' '
    for i6 in range(start, end + 1):
        file_name_for_5 = download_path_1 + '\\' + str(i6) + 'th_data.json'
        save_file_name_for_5 = download_path_5 + '\\' + str(i6) + 'th_data.csv'
        data = json.load(open(file_name_for_5))  # dict
        data_2 = data['data']  # list of dicts
        num = []
        raw_text = []
        for i66 in range(len(data_2)):
            num.append(data_2[i66]['publication_number'] + '_title')
            num.append(data_2[i66]['publication_number'] + '_abstract')
            num.append(data_2[i66]['publication_number'] + '_claims')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['title_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['abstract_localized'][0]['text']))
            except:
                raw_text.append('')
            try:
                raw_text.append(re.sub('[^%s]' % allow, "", data_2[i66]['claims_localized'][0]['text']))
            except:
                raw_text.append('')
        if (len(num) == len(raw_text) and len(num) % 3 == 0): print('doc ' + str(i6) + ' is well done')
        data_for_sao = pd.DataFrame({"num": num, "raw_text": raw_text})
        data_for_sao.to_csv(save_file_name_for_5, index=False, encoding='utf-8')

'''
sao_file_num=19
root_dir_ = download_path_6
ver_ = str(data_list_for_5[sao_file_num])
data_dir = download_path_5+'\\'+str(data_list_for_5[sao_file_num])
'''

def extract_SAO(root_dir_, ver_, data_dir):
    print("데이터 형태는 num raw_text 인 csv이여야 함")
    print("num_parsed_info.json 을 파일제목 parsing 결과 생성 및 다 합쳐서 sao추출결과 나옴 ")
    # 1. install and run coreNLP
    # https://github.com/hjzzang/study1_NLP/blob/master/05.%20installCoreNLP.md

    # 2. setting work directory and name of folder ver.
    '''
    root_dir = 'E:\\Dropbox\\☆★research\\☆2_patentinfringement\\3.data\\3rd_try\\0.data\\validation_real_company_which_related_with_nintendo'
    ver = '190814_'
    root_dir = 'C:\\Users\\ddonae\\Downloads'
    ver = '200705_'
    '''
    root_dir = root_dir_
    ver = ver_

    # 1. make directory with the name of 'adress' below
    adress = ver + 'parsing'
    os.mkdir(root_dir + "/" + adress + "/")
    os.chdir(root_dir)

    # raw patnet data (defalt file of wisdomain ver.)
    # data = pd.read_csv('data_B60_2018.csv', header=4, keep_default_na=False, encoding='cp949')
    """
    data1 = pd.read_csv('minoltacameracoltd.csv', header=4, keep_default_na=False)
    data2 = pd.read_csv('nintendo.csv', header=4, keep_default_na=False)
    data3 = pd.read_csv('roundrockresearch.csv', header=4, keep_default_na=False)
    data4 = pd.read_csv('sonyinteractiveentertainment.csv', header=4, keep_default_na=False)

    data = pd.concat([data1, data2, data3, data4], ignore_index= True)

    data.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    data.to_csv('merge.csv')
    """
    '''
    data = pd.read_csv('related_company_patents.csv', header=4, keep_default_na=False)
    data.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    '''
    data = pd.read_csv(data_dir, keep_default_na=True)
    # data.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    data=data.dropna(axis=0)
    """
    1. parsed_information
    """

    ###################
    start1 = datetime.now()
    print(start1)

    # preprocessing patent into sent_tokenize and parsing
    lemmatizer = WordNetLemmatizer()

    # range(len(data))
    for i in range(0, len(data)):
        try:
            if i % 10 == 0: print(i, "/", len(data))
            sent_id = []
            word_id = []
            raw = []
            lemma = []
            ctag = []
            tag = []
            head_id = []
            rel = []
            head_raw = []
            head_lemma = []
            head_pos = []
            json_by_patent = pd.DataFrame()
            patent_no = data.iloc[i].num
            data_input = data.raw_text.iloc[i]
            # print(i, "/", len(data))
            for j in range(len(sent_tokenize(data_input))):
                data_abs_sent = sent_tokenize(data_input)[j]
                dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
                parse, = dep_parser.raw_parse(data_abs_sent)
                txt = str(parse.to_conll(10))
                txt_split = txt.split('\n')
                for txt in txt_split:
                    word_info = txt.split('\t')
                    if (len(word_info)) == 10:
                        sent_id.append(j + 1)
                        word_id.append(word_info[0])
                        raw.append(word_info[1])
                        ctag.append(word_info[3])
                        tag.append(word_info[4])
                        head_id.append(word_info[6])
                        rel.append(word_info[7])
                        try:
                            lemma.append(lemmatizer.lemmatize(word_info[1].lower(),
                                                              word_info[4][0].lower()))  # 명사/형용사로 쓰이는 동사의 lemma 처리
                        except:
                            lemma.append(word_info[2])
                for k in head_id[len(head_raw):]:
                    try:
                        head_raw.append(raw[int(k) - 1 + n])
                        head_lemma.append(lemma[int(k) - 1 + n])
                        head_pos.append(tag[int(k) - 1 + n])
                    except:
                        head_raw.append(raw[int(k) - 1])
                        head_lemma.append(lemma[int(k) - 1])
                        head_pos.append(tag[int(k) - 1])

                n = len(head_raw)

            dependency_test = pd.DataFrame(
                {"SENTENCENO": sent_id, "WORD_NO": word_id, "RAW": raw, "LEMMA": lemma, "CTAG": ctag, "TAG": tag,
                 "HEAD_id": head_id, "HEAD_lemma": head_lemma, "HEAD_raw": head_raw, "HEAD_pos": head_pos, "REL": rel})
            file_name = "%s/%s_%s_parsed_info.json" % (adress, i + 1, patent_no)
            json_by_patent = json_by_patent.append(dependency_test, ignore_index=True)
            json_by_patent.to_json(file_name, orient='records')
        except:
            json_by_patent.to_json(file_name, orient='records')

    end1 = datetime.now()
    print(end1)
    print('1:', end1 - start1)

    #######################
    # 문장단위

    #######################
    def json_open(json_dir):
        with open(json_dir) as json_data:
            obj = json.load(json_data)
            df = pd.DataFrame(obj, columns=["SENTENCENO", "WORD_NO", "RAW", "LEMMA", "CTAG", "TAG", "REL", "HEAD_id",
                                            "HEAD_lemma", "HEAD_raw", "HEAD_pos"])
        return df

    """
    2. NP
    """
    start2 = datetime.now()
    print(start2)

    file_header = '_parsed_info.json'
    phrase_info_file_header = '_phrase_info.json'
    phrase_relation_file_header = '_phrase_relation.json'

    ###############

    id = []
    PAT_NO = []
    SENT = []
    CNT = []
    PHRASE = []
    POS = []
    HEAD = []
    HEAD_id = []
    REL = []
    WORD_ID = []
    TYPE = []

    phrase = ""
    pos_phrase = ""
    rel_phrase = ""
    id_set = ""
    type = ""

    # range(len(data))
    for i in range(len(data)):
        print(i, "/", len(data))
        try:
            target_patent = data.iloc[i].num
            file_name = str(i + 1) + "_" + str(target_patent) + "_parsed_info.json"
            file_name = adress + "/" + file_name
            patent_info = json_open(file_name)  # open json file of parsed dataframe for each patent

            for j in range(len(patent_info)):
                phrase_info_dir = adress
                # NP
                if (patent_info.HEAD_pos.iloc[j][0] == 'N' and (
                        patent_info.REL[j] == 'amod' or patent_info.REL[j] == 'compound') and patent_info.HEAD_id[j] > \
                    patent_info.WORD_NO[j]) or patent_info.TAG.iloc[j][0] == 'N':

                    json_dir = adress
                    phrase = phrase + " " + patent_info.LEMMA[j]
                    pos_phrase = pos_phrase + " " + patent_info.TAG[j][0]
                    rel_phrase = rel_phrase + " " + patent_info.REL[j]
                    id_set = id_set + " " + patent_info.WORD_NO[j]
                    type = "NP"
                elif len(phrase) > 0 and phrase.count(" ") - 1 == id_set.count(" ") - 1 == pos_phrase.count(
                        " ") - 1 == rel_phrase.count(" ") - 1:
                    id.append(i + 1)
                    PAT_NO.append(target_patent)
                    SENT.append(patent_info.SENTENCENO[j])
                    CNT.append(phrase.count(" "))
                    PHRASE.append(phrase[1:])
                    HEAD.append(phrase.split()[phrase.count(" ") - 1])
                    HEAD_id.append(id_set.split()[phrase.count(" ") - 1])
                    POS.append(pos_phrase[1:])
                    REL.append(rel_phrase[1:len(rel_phrase)])
                    WORD_ID.append(id_set[1:])
                    TYPE.append(type)

                    phrase = ""
                    pos_phrase = ""
                    rel_phrase = ""
                    id_set = ""
                    type = ""
        except:
            print(i, "is errorhoho")
    ANP_info_df = pd.DataFrame(
        {"id": id, 'patent_id': PAT_NO, "sent_id": SENT, "type": TYPE, "word_id": WORD_ID, "phrase": PHRASE, "POS": POS,
         "head": HEAD, "HEAD_id": HEAD_id, "CNT": CNT, "REL": REL})

    ANP_info_df.to_excel(ver + 'ANP.xlsx')
   
    time.sleep(10)

    end2 = datetime.now()
    print(end2)
    print('2:', end2 - start2)

    """
    5. SAO , 주어동사목적어만 단일단어
    """
    start5 = datetime.now()
    print(start5)
    # 능동형만

    ###############

    id = []
    PAT_NO = []
    SENT = []
    S = []
    S_id = []
    A = []
    A_id = []
    O = []
    O_id = []
    TYPE = []

    # range(len(data))
    for i in range(len(data)):
        if i % 10 == 0: print(i, "th data is working in 5", "/", len(data))
        try:
            # print(i,"/",len(data))
            target_patent = data.iloc[i].num
            file_name = str(i + 1) + "_" + str(target_patent) + "_parsed_info.json"
            file_name = adress + "/" + file_name
            patent_info = json_open(file_name)  # open json file of parsed dataframe for each patent

            st_in_cmp = list(set(list(
                patent_info.SENTENCENO)))  # size for length of sentence, if len of sentence = 3, st_in_cmp = [1,2,3]
            binded_phrase_info = pd.DataFrame()
            phrase_relation_info = pd.DataFrame()

            for st_no in st_in_cmp:
                st_info = patent_info[
                    patent_info.SENTENCENO == st_no]  # extract selected sentence (st_no) from dataframe
                st_info = st_info.reset_index(drop=True)  # reset index from 0 of dataframe (ex 48-74 -> 0-26)
                head_word_no_list = list(set(list(st_info.HEAD_id)))
                head_word_no_list = [int(no) for no in head_word_no_list]  # extract numerical number for head word
                head_word_no_list.sort()

                nmod_word_no_list = list(set(list(st_info[st_info.REL == "nmod"].WORD_NO)))
                nmod_word_no_list = [int(no) for no in nmod_word_no_list]
                nmod_word_no_list.sort()

                nsubjpass_word_no_list = list(set(list(st_info[st_info.REL == "nsubjpass"].WORD_NO)))
                nsubjpass_word_no_list = [int(no) for no in nsubjpass_word_no_list]
                nsubjpass_word_no_list.sort()

                s = ""
                a = ""
                o = ""

                s_id = ""
                a_id = ""
                o_id = ""

                # active
                for head_word_no in head_word_no_list:
                    head_word_no = str(head_word_no)
                    type = "a"

                    phrase_sources = st_info[st_info.HEAD_id == head_word_no]

                    sa_raw = phrase_sources[(phrase_sources.REL == 'nsubj') & (phrase_sources.CTAG != 'WDT')]
                    ao_raw = phrase_sources[(phrase_sources.REL == 'dobj') & (phrase_sources.CTAG != 'WDT')]

                    if len(sa_raw) * len(ao_raw) > 0:
                        s = sa_raw.LEMMA.iloc[0]
                        s_id = sa_raw.WORD_NO.iloc[0]
                        o = ao_raw.LEMMA.iloc[0]
                        o_id = ao_raw.WORD_NO.iloc[0]
                        a = list(set([sa_raw.HEAD_lemma.iloc[0]] + [ao_raw.HEAD_lemma.iloc[0]]))[0]
                        a_id = list(set([sa_raw.HEAD_id.iloc[0]] + [ao_raw.HEAD_id.iloc[0]]))[0]
                    else:
                        s = ""
                        s_id = ""
                        a = ""
                        a_id = ""
                        o = ""
                        o_id = ""

                    if (len(a) * len(s) > 0) | (len(a) * len(o) > 0):
                        S.append(s)
                        S_id.append(s_id)
                        A.append(a)
                        A_id.append(a_id)
                        O.append(o)
                        O_id.append(o_id)
                        id.append(i + 1)
                        PAT_NO.append(target_patent)
                        SENT.append(st_no)
                        TYPE.append(type)

                # passive 1 - acl+nmod+case(by) ex. 'data captured by camera'
                for nmod_word_no in nmod_word_no_list:
                    type = "s1"
                    nmod_word_no = str(nmod_word_no)

                    # (nmod의 head 명사 중 case : by - 명사) phrase extraction - RAW:'by', HEAD:s를 나타내는 명사 / HEAD --> s
                    s_raw = st_info[(st_info['HEAD_id'] == nmod_word_no) & (st_info['LEMMA'] == 'by') & (
                            st_info['HEAD_pos'] != 'WDT')]

                    if len(s_raw) > 0:
                        s = s_raw.HEAD_raw.iloc[0]
                        s_id = nmod_word_no  # nmod

                        ao_raw = st_info[
                            st_info['WORD_NO'] == st_info[(st_info['WORD_NO'] == nmod_word_no)].HEAD_id.iloc[0]]
                        a = ao_raw.LEMMA.iloc[0]
                        a_id = ao_raw.WORD_NO.iloc[0]
                        if ao_raw.HEAD_pos.iloc[0] != "WDT":
                            o = ao_raw.HEAD_lemma.iloc[0]
                            o_id = ao_raw.HEAD_id.iloc[0]
                        else:
                            o = ""
                    else:
                        s = ""
                        a = ""
                        o = ""

                    if (len(a) * len(s) > 0) | (len(a) * len(o) > 0):
                        S.append(s)
                        S_id.append(s_id)
                        A.append(a)
                        A_id.append(a_id)
                        O.append(o)
                        O_id.append(o_id)
                        id.append(i + 1)
                        PAT_NO.append(target_patent)
                        SENT.append(st_no)
                        TYPE.append(type)

                # passive 2
                for nsubjpass_no in nsubjpass_word_no_list:
                    type = 's2'
                    nsubjpass_no = str(nsubjpass_no)
                    sa_raw = st_info[(st_info['WORD_NO'] == nsubjpass_no)]
                    s = ""
                    s_id = ""
                    o = sa_raw.LEMMA.iloc[0]
                    o_id = sa_raw.WORD_NO.iloc[0]
                    a = sa_raw.HEAD_lemma.iloc[0]
                    a_id = sa_raw.HEAD_id.iloc[0]

                    if (len(a) * len(o) > 0):
                        S.append(s)
                        S_id.append(s_id)
                        A.append(a)
                        A_id.append(a_id)
                        O.append(o)
                        O_id.append(o_id)
                        id.append(i + 1)
                        PAT_NO.append(target_patent)
                        SENT.append(st_no)
                        TYPE.append(type)
        except:
            print(i, "is errorhoho")

    SAO_info_df = pd.DataFrame(
        {"id": id, 'patent_id': PAT_NO, "sent_id": SENT, "s_id": S_id, "s": S, "a_id": A_id, "a": A, "o_id": O_id,
         "o": O, "type": TYPE})
    SAO_info_df.to_excel(ver + 'SAO.xlsx')

    end5 = datetime.now()
    print(end5)
    print('5:', end5 - start5)
    time.sleep(10)
    """
    6. SAO_extended
    """
    start6 = datetime.now()
    print(start6)

    import numpy as np

    # NP = pd.read_excel(ver + 'ANP.xlsx', keep_default_na=False, encoding='cp949')
    # SAO = pd.read_excel(ver + 'SAO.xlsx', keep_default_na=False, encoding='cp949')
    NP = pd.read_excel(ver + 'ANP.xlsx', keep_default_na=False)
    SAO = pd.read_excel(ver + 'SAO.xlsx', keep_default_na=False)

    S_extended = []
    O_extended = []
    S_extended_id = []
    O_extended_id = []

    for i in range(len(SAO)):
        if i % 100 == 0: print(i, "th SAO is working")
        id = SAO.iloc[i].id
        sent_id = SAO.iloc[i].sent_id
        s_id = SAO.iloc[i].s_id
        o_id = SAO.iloc[i].o_id

        hj_NP = NP[NP.id == id]
        hj_NP = hj_NP[hj_NP.sent_id == sent_id]

        try:
            hj_s = hj_NP[hj_NP.HEAD_id == int(s_id)]
            s_ex = hj_s.phrase.iloc[0]
            s_num = hj_s.word_id.iloc[0]
        except:
            s_ex = ""
            s_num = ""

        try:
            hj_o = hj_NP[hj_NP.HEAD_id == int(o_id)]
            o_ex = hj_o.phrase.iloc[0]
            o_num = hj_o.word_id.iloc[0]
        except:
            o_ex = ""
            o_num = ""

        S_extended.append(s_ex)
        O_extended.append(o_ex)
        S_extended_id.append(s_num)
        O_extended_id.append(o_num)

    # 64.20525646209717

    SAO['s_extended'] = np.array(S_extended)
    SAO['o_extended'] = np.array(O_extended)
    SAO['s_extended_id'] = np.array(S_extended_id)
    SAO['o_extended_id'] = np.array(O_extended_id)

    SAO.to_excel(ver + 'SAO_extended.xlsx')

    end6 = datetime.now()
    print(end6)
    print(end6 - start6)

    data_sent = []
    save_patent_id = []
    save_sent_id = []

    # SAO2vec needs raw sentence please add in line 65??
    for i in range(0, len(data)):
        data_input = data.raw_text.iloc[i]
        for j in range(len(sent_tokenize(data_input))):
            data_sent.append(sent_tokenize(data_input)[j])
            save_patent_id.append(i + 1)
            save_sent_id.append(j + 1)
    save = pd.DataFrame({"id": save_patent_id, "sent_id": save_sent_id, "sentence": data_sent})
    save.to_excel(ver + 'sentence_tag.xlsx')

    data_join = pd.merge(SAO, save, on=['id', 'sent_id'])
    data_join.to_excel(ver + 'SAO_with_sentence.xlsx')

data_dir = "C:/Users/Administrator/Desktop/coding/SAO/CSV2402220014.csv"
root_dir = "C:/Users/Administrator/Desktop/coding/SAO/"
ver = "2023_0226_new"

extract_SAO(root_dir, ver, data_dir)