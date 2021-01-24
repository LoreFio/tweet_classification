# Sentiment Analysis

Code for the sentiment analysis pipeline

# how to install
pip install keras==2.3.1
#attention keras must be the 2.3.1 version
pip install glove
pip install nltk
pip install keras-self-attention
python -m spacy download en_core_web_md
python -m spacy download fr_core_news_md
python -m nltk.downloader all


git clone https://codehub.tki.la/lfiorentini/sentiment-analysis.git
cd sentiment-analysis/
mkdir proc_data

# how to execute for english

python3 create_long_mix.py
python3 preprocess.py -d "long_mix"
python3 LSTM_trained_embedding.py -d "long_mix"
python3 classify_script.py -f 'I am happy' 'I am sad'

# how to execute for french

python3 prepare_fr.py -p 0.4 #use -p 0.7 or higher for better results
#the parameters of the prepare_fr.py script allows the user to choose how big is the training set. Bigger training set imply better performances and much more time. This parameter has to me strictly bigger than 0 and lower than 1
python3 preprocess.py -d "fr"
python3 LSTM_trained_embedding.py -d "fr"
python3 classify_script_fr.py -f 'Je suis content' 'Je suis triste'
