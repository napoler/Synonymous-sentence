import sqlite3
from tqdm import tqdm
class Db:
    def __init__(self):
        self.dataDB ='data/data.db'
        self.conn = sqlite3.connect(self.dataDB)
        self.connect = self.conn.cursor()
        # self.connect =
    def close(self):
        """
        关闭数据库
        """
        self.conn.close()

    def create_table(self):
        # conn = sqlite3.connect(self.DB)
        # c = conn.cursor()
        # Create table
        #创建链接表

        self.connect.execute('''CREATE TABLE `nodes` ( `id` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, `text` TEXT NOT NULL )''')
        pass
   
    def add_nodes(self,nodes):
        """
        添加数据
        """
        sql="INSERT INTO nodes VALUES (?,?)"
        texts=[]
        for i,item in  tqdm(enumerate(nodes)):
            # print("".join(item['word_list']))
            texts.append((i,"".join(item['word_list'])+"\n"))
            if i%10000==0 :
                # print('10000')
                self.connect.executemany(sql,texts)
                self.conn.commit() 
                texts=[]
        self.connect.executemany(sql,[(i,"".join(item['word_list']))])
        self.conn.commit()  
    def get_node(self,id):
        """
        获取对应id的内容
        """
        sql= "select * from nodes where nodes.id="+str(id)
        # print(sql)
        self.connect.execute(sql)
        one=  self.connect.fetchone()
        return one
