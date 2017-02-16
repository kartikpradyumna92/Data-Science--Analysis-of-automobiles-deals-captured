<html>
<body>
<h2>Data Science project: Analyzing Automobile dealership data. </h2>
<h4>     This project aims in predicting the probability of capturing certain deals based on customer information and vehicle information. <hr><br> </h4>
<h3>
Description:
</h3>
<h4>
     Classes marked as 1 depict the deals that are captured and observations with class marked as 0 depict that the deals are not captured. The dataset is an imbalanced with majority of observations belonging to uncaptured deals (Class = 0). Also it contained few missing values and outliers which were dealth with during pre-processing stage.
     </h4>
     <br>
<h3>
Explanations: <br>
</h3>
<h4>
The result displays the stages of data preprocessing as well as implementation of Machine learning algorithm (Ensemble methods).<br>
&emsp; - Performed Feature Selection to remove all irrelevant features from the dataset. <br>
&emsp; - After the dimension is reduced, used IQR (Inter Quartile Range to identify and remove the outliers.
&emsp; - Used Bagging classifier technique (Ensemble method) to classify the data and further predict the probability of occurence of Class = 1 <br>
&emsp; - Bagging classifier provides good results with least misclassified observations for my imbalanced dataset.<br>
&emsp; - Calculated f1_score, area under curve and obtained confusion matrix to evaluate the working of my model. <br>
Original percentage of deal being captured is around 5.8%. My results come to around 6.3%.
</h4>
<hr>
<br>
<h3>Programming tools and Packages:<br></h3>
<h4>
I used PyCharm IDE with Anaconda 3 as project interpreter. <br>
&emsp; -> Anaconda 3 <br>
&emsp; -> scikit-learn <br>
&emsp; -> pandas <br>
&emsp; -> numpy <br>
</h4>
<br>
<h3>
<hr>
Results: </h3><br>
<h4>
&emsp; - f1_score: 0.905041605482 <br>
&emsp; - Confusion Matrix: <br>
&emsp;&emsp;&emsp;   0 &emsp;&emsp;1 <br>
  0 &emsp; 71809 &emsp; 6 <br>
  1 &emsp;   770 &emsp; 3698 <br>
&emsp; - Area under curve: 0.91378991803 <br>
&emsp; - Average probability of deals being Captured is:  0.0631150452918710
</h4>
</body>
</html>
