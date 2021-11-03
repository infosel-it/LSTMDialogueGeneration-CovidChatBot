import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers , activations , models , preprocessing

from tensorflow.keras import preprocessing , utils
import os
import yaml
import re

def prepareqa():
    dir_path ='./data'
    files_list = os.listdir(dir_path + os.sep)
    
    
    questions = list()
    answers = list()
    
    for filepath in files_list:
        stream = open( dir_path + os.sep + filepath , 'rb')
        docs = yaml.safe_load(stream)
        conversations = docs['conversations']
        for con in conversations:
            if len( con ) > 2 :
                questions.append(con[0])
                replies = con[ 1 : ]
                ans = ''
                for rep in replies:
                    ans += ' ' + rep
                answers.append( ans )
            elif len( con )> 1:
                questions.append(con[0])
                answers.append(con[1])
    
    answers_with_tags = list()
    for i in range( len( answers ) ):
        if type( answers[i] ) == str:
            answers_with_tags.append( answers[i] )
        else:
            questions.pop( i )
    
    answers = list()
    for i in range( len( answers_with_tags ) ) :
        answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )
    
    return questions,answers

questions, answers = prepareqa()
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( questions + answers )
VOCAB_SIZE = len( tokenizer.word_index )+1
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))

vocab = []
for word in tokenizer.word_index:
    vocab.append( word )

def tokenize( sentences ):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub( '[^a-zA-Z]', ' ', sentence )
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append( tokens )
    return tokens_list , vocabulary

def createlstmdialogmodel(): 

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences( questions )
    maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
    padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions , maxlen=maxlen_questions , padding='post' )
    encoder_input_data = np.array( padded_questions )
    print( encoder_input_data.shape , maxlen_questions )
    
    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences( answers )
    maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
    padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
    decoder_input_data = np.array( padded_answers )
    print( decoder_input_data.shape , maxlen_answers )
    
    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences( answers )
    for i in range(len(tokenized_answers)) :
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
    onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
    decoder_output_data = np.array( onehot_answers )
    print( decoder_output_data.shape )
    
    
    encoder_inputs = tf.keras.layers.Input(shape=( maxlen_questions , ))
    encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]
    
    decoder_inputs = tf.keras.layers.Input(shape=( maxlen_answers ,  ))
    decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax ) 
    output = decoder_dense ( decoder_outputs )
    
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    model.load_weights('model.h5')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')
    
    model.summary()

    return encoder_inputs,encoder_states,decoder_embedding,decoder_inputs,decoder_lstm,decoder_dense,maxlen_questions,maxlen_answers

#model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150) 
#model.save( 'model.h5' )

encoder_inputs,encoder_states,decoder_embedding,decoder_inputs,decoder_lstm,decoder_dense,maxlen_questions,maxlen_answers = createlstmdialogmodel()

def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model

def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')

enc_model , dec_model = make_inference_models()

def chatbot_response(msg):
    states_values = enc_model.predict( str_to_tokens(msg) )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]
    
    response = decoded_translation
    response = response.rsplit(' ', 1)[0]    
    return response


#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("LSTM Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="9", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
EntryBox.bind("<Return>", lambda x:send() )
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()