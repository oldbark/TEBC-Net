## TEBC-Net: An effective relation extraction approach for simple question answering over knowledge graphs

This is the code for the paper [TEBC-Net]. 

The optimization model used in this paper is the cnn.py provided above.


#### Installation
Requirements  
1. fuzzywuzzy  
2. scikit-learn  
3. torchtext  
4. nltk  
5. pytorch  
6. numpy

#### Trainning
we have integrated our TEBC-Net into a KGSQA framework:the Knowledge Embedding based Question Answering (KEQA) framework.Generally, the KEQA framework performs 3 main tasks: head entity recognition, relation extraction and head entity detection (HED). We use TEBC-Net to replace the original model in the KEQA framework to complete head entity
recognition and relation extraction respectively, and we adjust the output of TEBC-Net to replace the original model in KEQA to complete the HED.   

We use the following three statements to train the HED model, the entity extraction model and the relation extraction model, respectively.  
1. python train_detection.py  
2. python train_entity.py  
3. python train_pred.py  


#### Test
We use the following statement for the final KGSQA task test.  
python test_main.py

The specific model structure is commented inside the code.












