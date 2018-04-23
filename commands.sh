#For deep-crf char level feature extraction 
export KERAS_BACKEND=theano

nohup python getembeddings_generic.py --giza_prefix ../data/giza/actual.A3.final --segment_file ../data/segments.txt --map_file ../data/giza/actual.actual.ti.final --sentence ../data/train --sample_sent ../data/sample.txt --output_file ../data/giza/synthetic.txt > /tmp/nohup_giza.out &

#Attention
python -m nmt.getembeddings_generic --segment_file ../data/segments.txt --model_dir /tmp/nmt_model_en_hi_devnagari_wordembed_all/ --sentence_prefix ../data/train --sample_sent ../data/sample.txt --output_file /tmp/code_mixed_eval1/output_attention_all.csv


#Giza EMD
nohup python getembeddings_emd.py --giza_prefix ../data/giza/answer.A3.final --segment_file ../data/segments_new.txt --map_file ../data/giza/answer.actual.ti.final --sentence_prefix /tmp/nmt_data_en_hi/train --embedding_prefix ../data/embedding/embedding_new --sample_sent ../data/sample.txt --output_file /tmp/emnlp/candidate/giza_emd.tsv > /tmp/emnlp/candidate/nohup.out &

# Giza generic
nohup python getembeddings_generic.py --giza_prefix ../data/giza/answer.A3.final --segment_file ../data/segments_new.txt --map_file ../data/giza/answer.actual.ti.final --sentence /tmp/nmt_data_en_hi/train --sample_sent ../data/sample.txt --output_file /tmp/emnlp/candidate/giza_generic.tsv > /tmp/emnlp/candidate/nohup_gg.out &

#Get the POS of the english sentence
python RunModel.py ../../emnlp2017-bilstm-cnn-crf_bidisha/models/POS_old/POS/EN_UD_POS.h5 ../data/train_en_hi_new.txt ../data/POStagged_new.txt

#alignpos
python alignPOS.py --synthetic_file ../data/synthetic1.txt --align_file ../data/word_align/forward.align --pos_file ../data/POStagged.txt --out_file ../data/training.txt 


