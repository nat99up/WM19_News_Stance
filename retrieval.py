import json
import jieba
import pandas as pd
import numpy as np
import csv
from argparse import ArgumentParser
from langModel import Retrieval_LM,Query_expand


parser = ArgumentParser()
parser.add_argument("-i", "--inverted_file", default='inverted_file.json', dest = "inverted_file", help = "Pass in a .json file.")
parser.add_argument("-q", "--query_file", default='QS_1.csv', dest = "query_file", help = "Pass in a .csv file.")
parser.add_argument("-c", "--corpus_file", default='NC_1.csv', dest = "corpus_file", help = "Pass in a .csv file.")
parser.add_argument("-o", "--output_file", default='output/sample_output.csv', dest = "output_file", help = "Pass in a .csv file.")


args = parser.parse_args()

# Load inverted file
with open(args.inverted_file) as f:
    invert_file = json.load(f)
print('inverted file already loaded!')

# Get Vocabs 
terms = list(invert_file.keys())
doc_length_filename = 'DocLen.json'


# Read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
num_corpus = corpus.shape[0] # used for random sample



# Calculate documents length
document_length = dict()
prob_word = dict()

for i,(word,info) in enumerate(invert_file.items()):
    
    idf = info['idf']
    
    prob_word[word] = 0
    
    for document_count_dict in info['docs']:
        for doc, doc_tf in document_count_dict.items():
            
            # 紀錄document長度
            if doc not in document_length:
                document_length[doc] = doc_tf
            else:
                document_length[doc] += doc_tf
                
            # 紀錄此vocab出現次數
            prob_word[word] += 1
    
    if i % 1000 == 0:
        print('{} words has been processed!'.format(i))

# Calculate p(w|REF)
total_num = sum(prob_word.values())

for word,freq in prob_word.items():
    
    prob_word[word] = freq/total_num

print('確認總機率：%.2f' % sum(prob_word.values()))
print('document length already!')
print('p(w|REF) already!')



# Dirichlet Prior Parameter
u = 504 * 4.2


# Process each query
final_ans = []

for (query_id, query) in querys:
    
    
    print("query_id: {}".format(query_id))
          
    query_words = list(jieba.cut(query))
    
    retrieval_result = Retrieval_LM(query_word_list=query_words,
                                    vocab_list=terms,
                                    inv_list=invert_file,
                                    doc_len=document_length,
                                    prob_REF=prob_word,
                                    u=u)
    
    
    new_query_words = Query_expand(original_query_words=query_words,
                                   ret_res=retrieval_result,
                                   inv_list=invert_file,
                                   top_d=10,
                                   top_k=20)
    
    print('第%d次Feedback完成'%1)
    
    retrieval_result = Retrieval_LM(query_word_list=new_query_words,
                                    vocab_list=terms,
                                    inv_list=invert_file,
                                    doc_len=document_length,
                                    prob_REF=prob_word,
                                    u=u)
    
    new_query_words = Query_expand(original_query_words=new_query_words,
                                   ret_res=retrieval_result,
                                   inv_list=invert_file,
                                   top_d=30,
                                   top_k=40)
    
    print('第%d次Feedback完成'%2)
    
    retrieval_result = Retrieval_LM(query_word_list=new_query_words,
                                    vocab_list=terms,
                                    inv_list=invert_file,
                                    doc_len=document_length,
                                    prob_REF=prob_word,
                                    u=u)
    
    
    
    print('%s documents likelihood completed' % (query_id))
    
    final_ans.append(retrieval_result)

# Write answer to csv file
with open(args.output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    head = ['Query_Index'] + ['Rank_%03d'%i for i in range(1,301)]
    writer.writerow(head)
    for query_id, ans in enumerate(final_ans, 1):
        writer.writerow(['q_%02d'%query_id]+ans)
        

    








