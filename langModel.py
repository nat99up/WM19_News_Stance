#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 01:06:56 2019

@author: xiaohaoren
"""
import numpy as np
import operator
from collections import Counter


def Retrieval_LM(query_word_list,vocab_list,inv_list,doc_len,prob_REF,u = 2116):
    
    query_cnt = Counter()
    query_cnt.update(query_word_list)
    query_len = sum([len(c) for c in query_word_list ])
    
    document_scores = dict()
    document_cwd = dict() # c(w,d)
    
    # Speed up the indexing word operation(term.index(word))
    query_word_index = dict()
    for word in query_cnt.keys():
        try:
            query_word_index[word] = vocab_list.index(word)
        except:
            pass
    
    # Calculate c(w|d)
    # Initialize document scores
    for (word, count) in query_cnt.items():
        if word in query_word_index: #avoid the unseen term of inverted list
            
            term_idx = query_word_index[word]
            term_doc_dict = inv_list[word]['docs']
            
            for document_count_dict in term_doc_dict:        
                for doc, doc_tf in document_count_dict.items():            
                    if doc not in document_cwd:
                        document_cwd[doc] = np.zeros(len(vocab_list),np.int64)
                    # count c(w|d)
                    document_cwd[doc][term_idx] = doc_tf
                    # a_d
                    ad = u/(u + doc_len[doc])
                    # initialize the document score and normalize
                    if doc not in document_scores:
                        document_scores[doc] = query_len * np.log(ad)
                      
    # calculate p(w|d) by smoothing c(w,d)/|d| and p(w|REF)
    # calculate the document score by p(w|d) which w is appeared in query i
    for i , (doc , _) in enumerate(document_scores.items()):
        for word,count in query_cnt.items():
            
            if word not in query_word_index:
                continue
            
            term_idx = query_word_index[word]
            # c(w,d)/|d|
            cwd = document_cwd[doc][term_idx]
            # p(w|REF)
            pwREF = prob_REF[word]
            # calculate the document score
            document_scores[doc] += count * np.log(1 + cwd/(u * pwREF))
            
    
    
    # Sort the document score pair by the score
    sorted_document_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    # Take top 300 results
    res = [doc_score_tuple[0] for doc_score_tuple in sorted_document_scores[:300]]
    
    return res

def Query_expand(original_query_words,ret_res,inv_list,top_k = 3,top_terms = 10):
    

    
    feedback_doc = ret_res[:top_k]
    feedback_dict = dict()
    
    for (word,info) in inv_list.items():
        idf = info['idf']
        
        if idf < 10 or len(word)<2:
            continue
        for document_count_dict in info['docs']:
            for doc, doc_tf in document_count_dict.items(): 
                if doc in feedback_doc:
                    if word in feedback_dict:
                        feedback_dict[word] += doc_tf
                    else:
                        feedback_dict[word] = doc_tf
    
    feedback_lead = sorted(feedback_dict, key=feedback_dict.get, reverse=True)
    feedback_lead = feedback_lead[:top_terms]
    
    new_query_words = original_query_words + feedback_lead
    
    print('Exapended Query:',new_query_words)
        
    return new_query_words