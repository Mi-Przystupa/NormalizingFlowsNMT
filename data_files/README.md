# Data Files

These are the preprocessed files from the IWSLT dataset for de-en and en-de I used to evaluate model performance. You need to copy them into the .data/\<src\>-\<trg\> directory that gets created from TorchText in order to run the code.


For other language pairs, you need to create similarly named files of the following format:

- test.\<src\>-\<trg\>.\<src\>
- test.\<trg\>-\<src\>.\<trg\>
- val.\<src\>-\<trg\>.\<src\>
- val.\<trg\>-\<src\>.\<trg\>


In ../jupyter_notebooks/SentencePieceEncoding.ipynb there is the code I used to produce the files here. They might be useful for repeating the process on other language pairs. The function is merge_iwslt_bitext(...), where ... are the arguments. Requires jupyter and Python3 to run. 
