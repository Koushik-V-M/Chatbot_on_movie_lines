import numpy as np
import tensorflow as tf
import re
import time
import keras.layers as keras

lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')                   #getting the data in a list form from the text files
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
        
convo_id=[]
for line in conversations[:-1]:
    _convo=line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convo_id.append(_convo.split(','))                  #getting a list of all conversation id's
    
questions=[]
answers=[]
for id_list in convo_id:
    for id in range(len(id_list)-1):
        questions.append(id2line[id_list[id]])          #getting the conversations in questions and answers format
        answers.append(id2line[id_list[id+1]])

def clean(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)         #defining a function which cleans the text
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

clean_questions=[]
for question in questions:
    clean_questions.append(clean(question))
clean_answers=[]                                        #using the clean function to clean the question and answers
for answer in answers:
    clean_answers.append(clean(answer))

word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:                      
            word2count[word]=1
        else:
            word2count[word]+=1                         #searching the number of times a particualar word
for answer in clean_answers:                            #is used so we can delete the less used words
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

threshold=20
questionswords2int={}
word_num=0
for word,count in word2count.items():
    if count>=threshold:
        questionswords2int[word]=word_num
        word_num+=1                                     #deleting the terms which appear less then 20 times and
answerswords2int={}                                     #adding the rest to a dictionary with a unique integer
word_num=0
for word,count in word2count.items():
    if count>=threshold:
        answerswords2int[word]=word_num
        word_num+=1

tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:                                    #adding the last tokens to these two dictionaries
    questionswords2int[token]=len(questionswords2int)+1
for token in tokens:
    answerswords2int[token]=len(answerswords2int)+1

answersint2words={num:word for word,num in answerswords2int.items()}    #reversing the answerswords2int dictonary

for i in range(len(clean_answers)):
    clean_answers[i]+=' <EOS>'          #adding EOS to end of every sentence     

questions2int=[]
for sentence in clean_questions:
    inte=[]
    for word in sentence.split():
        if word not in questionswords2int:
            inte.append(questionswords2int['<OUT>'])
        else:                                           #converting all words in sentences of questions, answers
            inte.append(questionswords2int[word])       #into int using the questionswords2int,answerwords2int dictionaries
    questions2int.append(inte)
answers2int=[]
for sentence in clean_answers:
    inte=[]
    for word in sentence.split():
        if word not in answerswords2int:
            inte.append(answerswords2int['<OUT>'])
        else:
            inte.append(answerswords2int[word])
    answers2int.append(inte)

sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,26):
    for i in enumerate(questions2int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions2int[i[0]])
            sorted_clean_answers.append(answers2int[i[0]])

#creating a function to add placeholders for inputs and targets
def model_input():
    inputs=tf.placeholder(tf.int32, [None,None], name='input')
    target=tf.placeholder(tf.int32, [None,None], name='target')
    lr=tf.placeholder(tf.float32, name='learning_rate')
    keep_prob=tf.placeholder(tf.float32, name='keep_prob')
    return inputs,target,lr,keep_prob

def preprocess_targets(targets, word2int, batch_size):
    leftside=tf.fill([batch_size,1],word2int('<SOS>'))
    rightside=tf.strided_slice(targets, [0,0], [batch_size,-1],[1,1])
    preprocessed_targets=tf.concat([leftside,rightside],axis=1)
    return preprocessed_targets

def encoder_rnn(rnn_inputs,rnn_size,num_layers,keep_prob,sequential_length):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output,encoder_state=keras.layers.Bidirectional(keras.layers.RNN(encoder_cell))
    return encoder_state

def decoder_training_set(encoded_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,batch_size,keep_prob):
    attention_states=tf.zeros(batch_size,1,decoder_cell.output_size)
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states,attention_output="bahdanua",num_units=decoder_cell.output_size)
    training_decoder_function=tf.contrib.seq2seq.attention_decoding_fn_train(encoded_state[0],
                                                                                   attention_keys,
                                                                                   attention_values,
                                                                                   attention_score_function,
                                                                                   attention_construct_function,
                                                                                   name="attn_dec_train")
    decoder_output,_,_=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                              training_decoder_function,
                                                              decoder_embedded_input,
                                                              sequence_length,
                                                              scope=decoding_scope)
    decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return decoder_output_dropout

def decoder_test_set(encoded_state,decoder_cell,decoder_embeddings_matrix,sos_id,eos_id,max_length,num_words,sequence_length,decoding_scope,output_function,batch_size,keep_prob):
    attention_states=tf.zeros(batch_size,1,decoder_cell.output_size)
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states,attention_output="bahdanua",num_units=decoder_cell.output_size)
    test_decoder_function=tf.contrib.seq2seq.attention_decoding_fn_inference(encoded_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             decoder_embeddings_matrix,
                                                                             sos_id,
                                                                             eos_id,
                                                                             max_length,
                                                                             num_words,
                                                                             name="attn_dec_inf")
    test_predictions,_,_=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                              test_decoder_function,
                                                              scope=decoding_scope)
    return test_predictions

def decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,encoder_state,num_words,sequence_length,rnn_size,num_layers,word2int,keep_prob,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_words)
        weights=tf.truncated_normal_initializer(stddev=0.1)
        biases=tf.zeros_initializer()
        output_function=lambda x: tf.contrib.layers.fully_connected(x,
                                                                    num_words,
                                                                    None,
                                                                    weights,
                                                                    biases)
        training_predictions=decoder_training_set(encoder_state,
                                                  decoder_cell,
                                                  decoder_embedded_input,
                                                  sequence_length,
                                                  decoding_scope,
                                                  output_function,
                                                  batch_size,
                                                  keep_prob)
        decoding_scope.reuse_variables()
        testing_predictions=decoder_test_set(encoder_state,
                                             decoder_cell,
                                             decoder_embeddings_matrix,
                                             word2int['<SOS>'],
                                             word2int['<EOS>'],
                                             sequence_length-1,
                                             num_words,
                                             sequence_length,
                                             decoding_scope,
                                             output_function,
                                             batch_size,
                                             keep_prob)
    return training_predictions,testing_predictions

def seq2seq_model(inputs,targets,keep_prob,batch_size,sequential_length,answer_num_words,question_num_words,encoder_embedding_size,decoder_embedding_size,rnn_size,num_layers,questionswords2int):
    encoder_embedded_input=tf.contrib.layers.embed_sequence(inputs,
                                                     answer_num_words+1,
                                                     encoder_embedding_size,
                                                     initializer=tf.random_uniform_initializer(0,1))
    encoder_state=encoder_rnn(encoder_embedded_input,rnn_size,num_layers,keep_prob,sequence_length)
    preprocessed_targets=preprocess_targets(targets,questionswords2int,batch_size)
    decoder_embeddings_matrix=tf.Variable(tf.random_uniform([question_num_words+1,decoder_embedding_size],0,1))
    decoder_embedded_input=tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    training_predictions,testing_predictions=decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         question_num_words,
                                                         sequential_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions,testing_predictions


epochs=100
batch_size=64
rnn_size=512
num_layers=3
encoder_embedding_size=512
decoder_embedding_size=512
learning_rate=0.01
learning_rate_decay=0.09
min_learning_rate=0.0001
keep_probability=0.5

tf.reset_default_graph()
tf.InteractiveSession()

inputs,targets,lr,keep_prob=model_input()

sequence_length=tf.placeholder_with_default(inputs,None,name="sequence_length")

input_shape=tf.shape(inputs)

training_predictions,testing_predictions=seq2seq_model(tf.reverse(inputs,[-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoder_embedding_size,
                                                       decoder_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)

with tf.name_scope("optimization"):
    loss_error=tf.contrib.seq2seq.sequence_loss(training_predictions,targets,
                                               tf.ones([input_shape[0],sequence_length]))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
    gradients=optimizer.compute_gradients(loss_error)
    clipped_gradients=[(tf.clip_by_value(gradient_tensor,-5.,5.),gradient_value)for gradient_tensor,gradient_value in gradients if gradient_tensor is not None]
    clipped_optimizer=optimizer.apply_gradients(clipped_gradients)

def apply_padding(batch_of_sequences,word2int):
    max_length=max([len(sequence) for sequence in batch_of_sequences])
    return ([sequence + [word2int["<PAD>"]]*(max_length-len(sequence)) for sequence in batch_of_sequences])

def split_into_batchs(questions,answers,batch_size):
    for batch_index in range(0,len(questions)//batch_size):
        start_index=batch_size*batch_index
        questions_in_batch=questions[start_index,start_index+batch_size]
        answers_in_batch=answers[start_index,start_index+batch_size]
        padded_questions=np.array(apply_padding(questions_in_batch,questions2int))
        padded_answers=np.array(apply_padding(answers_in_batch,answers2int))
        yield padded_questions,padded_answers

training_validation_index=int(len(sorted_clean_questions)*0.15)
training_questions=sorted_clean_questions[training_validation_index:]
training_answers=sorted_clean_answers[training_validation_index:]
validation_questions=sorted_clean_questions[:training_validation_index]
validation_answers=sorted_clean_answers[:training_validation_index]

batch_check_tarining_loss=100
batch_index_check_validation_loss=((len(training_questions)// batch_size)//2)-1
total_training_loss_error=0
list_validations_loss_error=[]
early_stopping_check=0
early_stopping_stop=1000
checkpoint="chatbot_weights.ckpt"
session.run(tf.global_normal_initializer())
for epoch in epochs:
    for batch_index,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batchs(training_questions,training_answers,batch_size)):
        start_time=time.time()
        _,batch_train_loss_error=session.run([clipped_optimizer,loss_error],{inputs:padded_questions_in_batch,
                                                                             targets:padded_answers_in_batch,
                                                                             lr:learning_rate,
                                                                             sequence_length:padded_answers_in_batch.shape[1],
                                                                             keep_prob:keep_probability})
        total_training_loss_error+=batch_train_loss_error
        end_time=time.time()
        batch_time=end_time-start_time
        if batch_index % batch_check_tarining_loss == 0:
            print("epoch:{:>3}/{}, batch:{:>4}/{}, train_loss_error:{:>6.3f}, training_time_on_100_batchs:{:d}".format(epoch,
                                                                                                                       epochs,
                                                                                                                       batch_index,
                                                                                                                       len(training_questions)//batch_size,
                                                                                                                       total_training_loss_error/batch_check_tarining_loss,
                                                                                                                       int(batch_time*batch_check_tarining_loss)))
            total_training_loss_error=0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index!=0:
            total_validation_loss_error=0
            start_time=time.time()
            for batch_index_validation,(padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batchs(validation_questions,validation_answers,batch_size)):
                start_time=time.time()
                batch_validation_loss_error=session.run(loss_error,{inputs:padded_questions_in_batch,
                                                                    targets:padded_answers_in_batch,
                                                                    lr:learning_rate,
                                                                    sequence_length:padded_answers_in_batch.shape[1],
                                                                    keep_prob:1})
                total_validation_loss_error+=batch_validation_loss_error
            end_time=time.time()
            batch_time=end_time-start_time
            avg_validation_loss=total_validation_loss_error/(len(validation_questions)/batch_size)
            print("validation_loss_error={:>6.3f}, batch_validation_time:{:d} seconds".format(avg_validation_loss,int(batch_time)))
            learning_rate*=learning_rate_decay
            if learning_rate<min_learning_rate:
                min_learning_rate=learning_rate
            list_validations_loss_error.append(avg_validation_loss)
            if avg_validation_loss<min(list_validations_loss_error):
                print("i speak better now")
                early_stopping_check=0
                saver=tf.train.Saver()
                saver.save(session,checkpoint)
            else:
                print("i will practice more")
                early_stopping_check+=1
                if early_stopping_check==early_stopping_stop:
                    break
    if early_stopping_check==early_stopping_stop:
        print("sorry,this is the best i can do")
        break
print("game_over")










