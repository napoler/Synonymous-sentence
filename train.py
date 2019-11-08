import Terry_toolkit as tkit
import jieba
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument,TaggedLineDocument
from tqdm import tqdm
import os
import sys
import argparse
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
            os.remove('data/dataset/data.json')
        except:
            pass
    dataset=tkit.Json(file_path='data/dataset/data.json')
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
    if os.path.exists('data/dataset/data.json'):
        data=tkit.Json(file_path='data/dataset/data.json').auto_load()
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
    
    if os.path.exists('data/dataset/data.json'):
        data=tkit.Json(file_path='data/dataset/data.json').auto_load()
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

def train(documents, size=5000, epoch_num=1):
    """
    执行训练
    """
    print("开始训练")
    tkit.File().mkdir('model')
    # model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model = Doc2Vec(documents, vector_size=size, window=2, min_count=2, workers=4, epochs=epoch_num)
    model.save('model/doc2vec.model')
    return model
def train_epoch(documents, size=200, epoch_num=1):
    """
    进行迭代
    """
    # model = Doc2Vec(documents, vector_size=size, window=2, min_count=1, workers=4)
    # model.load('model/doc2vec.model')
    model = Doc2Vec.load("model/doc2vec.model")
    for i in range(epoch_num):
        model.train(documents, total_examples=model.corpus_count, epochs=epoch_num)
        model.save('model/doc2vec.model')



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

def run_train_epoch():
    documents=pre_train()
    train_epoch(documents)   

def main():
    parser = argparse.ArgumentParser(usage="运行训练.", description="help info.")
    # parser.add_argument("--address", default=80, help="the port number.", dest="code_address")
    # parser.add_argument("--flag", choices=['.txt', '.jpg', '.xml', '.png'], default=".txt", help="the file type")
    parser.add_argument("--do", type=str, required=True, help="输入运行的类型  (pre,train,train_epoch,build_dataset)")
    # parser.add_argument("-l", "--log", default=False, action="store_true", help="active log info.")
 
    args = parser.parse_args()
    # print("--address {0}".format(args.code_address))    #args.address会报错，因为指定了dest的值
    # print("--flag {0}".format(args.flag))   #如果命令行中该参数输入的值不在choices列表中，则报错
    print("--do {0}".format(args.do))   #prot的类型为int类型，如果命令行中没有输入该选项则报错
    # print("-l {0}".format(args.log))  #如果命令行中输入该参数，则该值为True。因为为短格式"-l"指定了别名"--log"，所以程序中用args.log来访问
    if args.do=='pre':
        pre()
    elif args.do=='train':
        run_train()
    elif args.do=='train_epoch':
        run_train_epoch()
    elif args.do=='build_dataset':
        build_dataset()
    elif args.do=='auto':
        pre()
        run_train()
        run_train_epoch()

 
if __name__ == '__main__':
    main()
