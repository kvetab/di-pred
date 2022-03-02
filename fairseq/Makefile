
### data directory at root
DATA_DIR := ../data/

### Tokenize raw sequences ###
TOKENIZE_RAW_DATA := ./tokenize_raw_sequences.py
TOKENIZE_LABELS := ./process_labels.py

### Train

data/raw/heavy/train: data/raw/heavy/train/train.in data/raw/heavy/train/train.out

data/raw/heavy/train/train.in: $(DATA_DIR)chen/deduplicated/chen_train_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 1
    
data/raw/light/train: data/raw/light/train/train.in data/raw/light/train/train.out

data/raw/light/train/train.in: $(DATA_DIR)chen/deduplicated/chen_train_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 2

# Labels

data/raw/heavy/train/train.out: $(DATA_DIR)chen/deduplicated/chen_train_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_LABELS) --input_data "$<" --out_data $@ 
    
data/raw/light/train/train.out: $(DATA_DIR)chen/deduplicated/chen_train_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_LABELS) --input_data "$<" --out_data $@ 


### Valid

data/raw/heavy/valid: data/raw/heavy/valid/valid.in data/raw/heavy/valid/valid.out

data/raw/heavy/valid/valid.in: $(DATA_DIR)chen/deduplicated/chen_valid_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 1
    
data/raw/light/valid: data/raw/light/valid/valid.in data/raw/light/valid/valid.out

data/raw/light/valid/valid.in: $(DATA_DIR)chen/deduplicated/chen_valid_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 2

# Labels

data/raw/heavy/valid/valid.out: $(DATA_DIR)chen/deduplicated/chen_valid_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_LABELS) --input_data "$<" --out_data $@ 
    
data/raw/light/valid/valid.out: $(DATA_DIR)chen/deduplicated/chen_valid_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_LABELS) --input_data "$<" --out_data $@ 



### Test

data/raw/heavy/test: data/raw/heavy/test/test.in data/raw/heavy/test/test.out

data/raw/heavy/test/test.in: $(DATA_DIR)chen/deduplicated/chen_test_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 1
    
data/raw/light/test: data/raw/light/test/test.in data/raw/light/test/test.out

data/raw/light/test/test.in: $(DATA_DIR)chen/deduplicated/chen_test_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 2

# Labels

data/raw/heavy/test/test.out: $(DATA_DIR)chen/deduplicated/chen_test_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_LABELS) --input_data "$<" --out_data $@ 
    
data/raw/light/test/test.out: $(DATA_DIR)chen/deduplicated/chen_test_data.csv
	mkdir -p $(dir $@)
	python $(TOKENIZE_LABELS) --input_data "$<" --out_data $@ 
    
    
    
### Binarize data ###

data/processed: data/processed/heavy/train data/processed/light/train


# 
data/processed/heavy/train: data/raw/heavy/train data/raw/heavy/valid data/raw/heavy/test data/raw/vocab.txt
	mkdir -p $(dir $@)
	fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --testpref $(word 3,$^)/test.in \
        --srcdict $(word 4,$^) \
        --only-source \
        --workers 2 \
        --destdir $@/input0
	fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --testpref $(word 3,$^)/test.out \
        --only-source \
        --workers 2 \
        --destdir $@/label
        
data/processed/light/train: data/raw/light/train data/raw/light/valid data/raw/light/test data/raw/vocab.txt
	mkdir -p $(dir $@)
	fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --testpref $(word 3,$^)/test.in \
        --srcdict $(word 4,$^) \
        --only-source \
        --workers 2 \
        --destdir $@/input0
	fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --testpref $(word 3,$^)/test.out \
        --only-source \
        --workers 2 \
        --destdir $@/label




### Train models ###

models/heavy/01_pretrained_2000epochs: data/processed/heavy/train/ /Users/brazdilk/Projects/BioPhi/biophi/humanization/methods/sapiens/models/v1/checkpoint_vh.pt
	mkdir -p $@
	fairseq-train \
		$< \
		--user-dir /Users/brazdilk/Projects/BioPhi/biophi/humanization/methods/sapiens \
		--init-token 0 --separator-token 2 \
		--restore-file $(word 2,$^) \
		--reset-optimizer --reset-dataloader --reset-meters \
		--save-dir $@/checkpoints \
		--arch roberta_small \
		--criterion sentence_prediction \
		--task sentence_prediction \
		--num-classes 2 \
		--optimizer adam \
		--lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
		--dropout 0.1 --attention-dropout 0.1 \
		--max-positions 144 \
		--shorten-method truncate \
		--batch-size 256 \
		--max-epoch 2000 \
		--log-format simple \
		--log-interval 1000 \
		--validate-interval 1 \
		--save-interval 100 \
			2>&1 | tee $@/log