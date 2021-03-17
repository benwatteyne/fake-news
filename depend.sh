#!/bin/bash 
pip3 install --user bs4==0.0.1
pip3 install --user editdistance==0.5.3
pip3 install --user gensim==3.8.1
pip3 install --user lime==0.2.0.1
pip3 install --user matplotlib==3.0.3
pip3 install --user nltk==3.4.5
pip3 install --user numpy==1.17.0
pip3 install --user pandas==0.25.1
pip3 install --user requests==2.22.0
pip3 install --user scipy==1.4.1
pip3 install --user seaborn==0.9.0
pip3 install --user tldextract==2.2.2
pip3 install --user torch==1.5.1
pip3 install --user tqdm==4.40.0
pip3 install --user jupyter
pip3 install --user jupyterlab
#python3 -m nltk.downloader all
export PATH="~/.local/bin:$PATH"
jupyter nbextension enable --py widgetsnbextension
#jupyter labextension install @jupyter-widgets/jupyterlab-manager

export NLTK_DATA=~/nltk_data/
tldextract --update

