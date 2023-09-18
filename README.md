# ReceiptClassification
A playground for my engineering project, I am trying to learn as much technlogies with this project as possible (hence for example i used YOLO for spliting receipt in to logo,top,products,bottom and tottal)

Overall idea for Pipeline:
1.Split Receipt with YOLO and take products only
2.Run OCR pipeline on products:
a)Adaptive Binarization
b)Component Analysis
c)Words/Lines Detection
d)Run OCR on every single line
3.Convert products from strings to embeddings
4.Classify based on embedings

