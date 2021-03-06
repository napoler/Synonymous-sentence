import Terry_toolkit as tkit
import jieba
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument,TaggedLineDocument
from tqdm import tqdm
import os
import sys
import argparse
import sqlite3
from db import Db
def pre_build_dataset (path='data',typen='new'):
    """
    构建训练数据
    预处理训练数据
    """
    # documents=[]
    f_list=tkit.File().all_path(path)
    tkit.File().mkdir('data/dataset')
    if typen == 'new':
        try:
            os.remove(DATA_FILE)
        except:
            pass
    dataset=tkit.Json(file_path=DATA_FILE)
    print("构建训练数据")
    for file_path in tqdm(f_list):
        documents=[]
        data=tkit.Json(file_path=file_path).auto_load()
        # print(data[:2])
        t_text=tkit.Text()
        seqs_all=[]
        for item in data:
            # print(item['text'])
            seqs= t_text.sentence_segmentation_v1(item['text'])
            seqs_all=seqs_all+seqs
        # print(len(seqs_all))
        # print(seqs_all[:10])

        seqs_words=[]
        for item in seqs_all:
            words=[]
            for word in jieba.cut(item):
                words.append(word)
            seqs_words.append(words)
        # print(seqs_words[:10])
        # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

        for word_list in  seqs_words:
            tag=' '.join(word_list)
            documents.append({'word_list':word_list})

            # document = TaggedDocument(word_list, tags=[tag])
            # documents.append(document)
        dataset.save(documents)
    print('预处理结束')
    return True


def build_dataset():
    """
    读取预处理的数据集
    """
    if os.path.exists(DATA_FILE):
        data=tkit.Json(file_path=DATA_FILE).auto_load()
        documents=[]
        print('build_dataset')
        for i,item in tqdm(enumerate(data)):
            document = TaggedDocument(item['word_list'], tags=[i])
            # yield TaggedDocument(item['word_list'], tags=[i])
            documents.append(document)
        return documents
    else:
        return []


def data_txt():
    
    if os.path.exists(DATA_FILE):
        data=tkit.Json(file_path=DATA_FILE).auto_load()
        # documents=[]
        print('data_txt')
        # f=open('mergeTXT.txt','a+')
        f=open('data/dataset/data.txt','w')
        for i,item in tqdm(enumerate(data)):
            new_context = " ".join(item['word_list']) + '\n'
            f.write(new_context)
        f.close()
        return True
    else:
        print('data.json不存在')
        return False
def pre_train():
    if os.path.exists('data/dataset/data.txt'):
        # documents = TaggedLineDocument('data/dataset/data.txt')
        print("data.txt 文件存在")
        pass
    else:
        data_txt()
    documents = TaggedLineDocument('data/dataset/data.txt')
    return documents


def file_List(self, path, type='txt'):
    files = []
    for file in os.listdir(path):
        if file.endswith("." + type):
            print(path+file)
            files.append(path+file)
    return files

def train(documents, size=50, epoch_num=1):
    """
    执行训练
    """
    print("开始训练")
    tkit.File().mkdir('model')
    # model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model = Doc2Vec(documents, vector_size=size, window=2, min_count=2, workers=4, epochs=epoch_num)
    model.save('model/doc2vec.model')
    return model
def train_epoch(documents,epoch_num=1):
    """
    进行迭代
    """
    # model = Doc2Vec(documents, vector_size=size, window=2, min_count=1, workers=4)
    # model.load('model/doc2vec.model')
    model = Doc2Vec.load(MODEL_FILE)
    for i in tqdm(range(epoch_num)):
        print("epoch",i)
        # tte = 37              #total_examples参数更新
        # model.train(documents, total_examples=tte, epochs=epoch_num)
        model.train(documents, total_examples=model.corpus_count, epochs=epoch_num)
        # model.save('model/tmp_doc2vec.model')
    model.save('model/doc2vec.model')


def test(text=''):
    model_dm = Doc2Vec.load(MODEL_FILE)
    text_cut = jieba.cut(text)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims


# # #预处理训练数据
# # pre_build_dataset(path='data/data/wiki_zh/')
# # #加载预处理数据
# documents=build_dataset()

# print(len(documents))
# # train(documents)

# train_epoch(documents)
 
def pre():
    pre_build_dataset(path='data/data/wiki_zh/')
# def build_dataset():
#     build_dataset()

def run_train():
    print('train')
    documents=pre_train()
    # print(len(documents),'条训练数据')
    print('start train')

    train(documents)

def run_train_epoch(epoch=10):
    documents=pre_train()
    train_epoch(documents=documents,epoch_num=epoch)   

def run_test(text):
    print('test 获取预测结果')
    sims = test(text)
    # print(sims)
    db_s=False
    if os.path.exists('data/data.db'):  
        db=Db()
        db_s=True
        print('已经索引数据库,可以直接获取相关文字信息 构建索引 运行" python3 train.py --do db" ')
    p=[]
    for tid, sim in sims:
        # print(item)

        if db_s:
            text=db.get_node(tid)
            p.append((tid,sim,text[1]))
            # print(tid,sim)
        else:
            # print(tid,sim)
            p.append((tid,sim))
            pass
    print(p)
    return p



def run_db():
    # run_db_init()
    if os.path.exists('data/data.db'):   
        # os.remove("data/data.db")
        print('data/data.db已经存在 请手动删除')
        return 
    
    db=Db()
    # if os.path.exists('data/data.db'):
    db.create_table()

    # if os.path.exists('data/data.db'):
    # db.create_table()
    if os.path.exists(DATA_FILE):
        data=tkit.Json(file_path=DATA_FILE).auto_load()
        # for item in data:
            # print(item)
        db.add_nodes(data)
 
    pass
def run_db_init():
    if os.path.exists('data/data.db'):   
        os.remove("data/data.db")
    db=Db()
    # if os.path.exists('data/data.db'):
    db.create_table()
    
    pass
def main():
    parser = argparse.ArgumentParser(usage="运行训练.", description="help info.")
    # parser.add_argument("--address", default=80, help="the port number.", dest="code_address")
    # parser.add_argument("--flag", choices=['.txt', '.jpg', '.xml', '.png'], default=".txt", help="the file type")
    parser.add_argument("--do", type=str, required=True, help="输入运行的类型  (pre,train,train_epoch,build_dataset,data_txt,test,db )")
    # parser.add_argument("-l", "--log", default=False, action="store_true", help="active log info.")
    parser.add_argument("--text", type=str, required=False, help="输入文本")
    parser.add_argument("--epoch", type=int, required=False, help="运行迭代的次数")
    parser.add_argument("--file", type=str,default= 'data/dataset/data.json', required=False, help="自定义训练数据")
    parser.add_argument("--model", type=str,default= 'model/doc2vec.model', required=False, help="输入之前的模型")
    # parser.add_argument("--out", type=str,default= 'model/doc2vec.model', required=False, help="输出模型的位置")
    args = parser.parse_args()
    # print("--address {0}".format(args.code_address))    #args.address会报错，因为指定了dest的值
    # print("--flag {0}".format(args.flag))   #如果命令行中该参数输入的值不在choices列表中，则报错
    print("--do {0}".format(args.do))   #prot的类型为int类型，如果命令行中没有输入该选项则报错
    # print("-l {0}".format(args.log))  #如果命令行中输入该参数，则该值为True。因为为短格式"-l"指定了别名"--log"，所以程序中用args.log来访问
    global MODEL_FILE
    global DATA_FILE
    global MODEL_OUT
    MODEL_FILE=args.model
    DATA_FILE=args.file
    # MODEL_OUT=args.out
    if args.do=='pre':
        pre()
    elif args.do=='train':
        run_train()
    elif args.do=='train_epoch':
        run_train_epoch(epoch=args.epoch)
    elif args.do=='build_dataset':
        build_dataset()
    elif args.do=='data_txt':
        data_txt()
    elif args.do=='test':
        run_test(args.text)
    elif args.do=='db':
        run_db()
    elif args.do=='auto':
        pre()
        run_train()
        run_train_epoch()

# DATA_FILE= 'data/dataset/data.json'
if __name__ == '__main__':
    main()
