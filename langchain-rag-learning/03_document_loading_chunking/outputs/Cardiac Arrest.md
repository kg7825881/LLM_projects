- 1 **Visual Analysis of Cardiac Arrest Prediction Using Machine** 

- 2 **Learning Algorithms: A Health Education Awareness Initiative** 

## 3 

# **ABSTRACT** 

4 A visual analysis may accurately predict cardiac arrest, making it a potent educational tool for raising 5 public awareness of health issues. By predicting cardiac arrest earlier, preventative steps can be taken 6 to save lives and the dissemination of such health knowledge can dramatically lower the world 7 mortality rate. A heart attack, also known as cardiac arrest, encompasses various heart-related 8 disorders and has been the leading cause of death worldwide in recent decades. Several medical data 9 mining and machine learning technologies are being applied to gather helpful knowledge regarding 10 heart disease prediction. The accuracy of the intended outcomes, however, is insufficient. This 11 chapter aims to predict the likelihood of patients having heart disease to solve the issue. Specifically, 12 it compared alternative models for the identification of cardiac arrest to appropriately categorize and 13 forecast heart attack instances with compact features. The use of ensemble algorithms over classifier 14 algorithms gives a maximum accuracy of 96.5%, which is examined in our investigation. 

- 15 **Keywords** : Cardiac Arrest, Machine Learning, Comparative Analysis, Heart Disease Prediction, 16 Data Standardization, Supervised Learning, Health Education Awareness 

## 17 

# **INTRODUCTION** 

18 Visual analysis is an effective technique for educating society. Raising public awareness on various 19 health issues can lead to taking preventative actions. For instance, through earlier cardiac arrest 20 prediction, curative measures can be taken in advance to save human life (Maaliw, Alon, Lagman, 21 Garcia, Susa, et al., 2022). Promoting such health knowledge in society can dramatically lower the 22 world mortality rate. According to the World Health Organization (2021), cardiovascular diseases 23 (CVDs) are the leading cause of mortality globally, killing an estimated 17.9 million people each 24 year. Heart attacks and strokes are responsible for 85% of these fatalities. CVDs include coronary 25 heart disease, cerebrovascular illness, rheumatic heart disease, and other diseases (Amini et al., 26 2021). Nearly three-quarters of all heart-related fatalities occur in low- and middle-income nations. 27 In 2015, low- and middle-income nations accounted for 82% of the 17 million early deaths (before 28 the age of 70) related to non-communicable illnesses, accounting for a total of 37%. Four out of 29 every five CVD fatalities are caused by heart attacks or strokes, among them, one-third of the deaths 30 occur in those below 70-year old patients (Desai et al., 2021). High blood pressure, cholesterol, and 31 lipid levels, as well as being overweight or obese, are all signs of heart illness (Jurgens et al., 2022). 32 Identifying patients who are more likely to suffer from cardiac arrest and ensuring that they receive 33 proper care would assist to prevent early deaths. All primary health care providers should give access 34 to important noncommunicable disease drugs and basic health technologies to ensure that individuals 35 in need receive treatment and counselling. Most heart attacks can be averted by addressing various 36 behavioural risk factors such as cigarette use (Sandhu et al., 2012), poor nutrition and obesity (Garcia 37 & Garcia, 2023), problematic alcohol consumption (Chudzińska et al., 2022), and lack of physical 38 inactivity through population-wide initiatives. 

58 

59 

- 39 Any serious cardiac ailment is referred to as cardiovascular disease. Because they can be fatal, 40 researchers are concentrating on developing smart systems to precisely identify cardiac illnesses 41 based on electronic health data, with the use of machine learning algorithms (Pal et al., 2022). For 

- 42 instance, Javeed et al. (2022) investigated several machine learning techniques for heart disease 43 prediction that make use of patient data on key health indicators. Over the past few decades, 44 cardiovascular illnesses have become the leading cause of mortality globally in both industrialized 45 and developing nations (Bae et al., 2021; Ruan et al., 2018). The mortality rate can be decreased 46 through early identification of heart disorders and ongoing clinical monitoring by professionals. 47 However, because it takes more intelligence, effort, and knowledge, reliable identification of cardiac 48 problems in all situations and 24-hour patient consultation by a doctor are not yet possible. 

- 49 Thanks to the growing volume of data, machine learning is becoming more popular. Machine 50 learning allows humans to gain information from vast amounts of data, which is difficult and often 51 impractical for humans to do. Furthermore, and perhaps most importantly, intelligent optimization 52 algorithms are used to compare the various machine learning techniques as stated in Figure 1. 53 Throughout the world, machine learning is applied in numerous fields (Garcia et al., 2019; Maaliw, 54 Susa, et al., 2022). No industry is more so than the medical field. Machine learning has the potential 55 to be extremely useful in determining whether certain conditions, such as heart disease and 

- 56 locomotor disorders, will exist. A doctor may be able to make substantial adjustments to their 57 diagnosis and course of therapy if such information is anticipated well in advance. 



**Figure 1. Broad Classification of Machine Learning Techniques** 

# 60 **MAIN FOCUS OF THE CHAPTER** 

- 61 In this chapter, we did an exhaustive comparison and assessment of several classifiers for cardiac 62 arrest prediction using three distinct baskets: bagging methods, boosting algorithms, and supervised 63 algorithms. Worldwide, heart disease has been identified as a leading cause of death. Early disease 64 detection is more crucial and valuable, and Auto-Prognosis significantly improved cardiac threat 

- 65 assessment performance when compared to well-performing systems. Emulation of the decision- 

- 66 making system that can identify cardiovascular illness using binary classification algorithms is part 

67 of the architecture of an intelligent system. An intelligent system attempts to identify the disease 68 using human indications as inputs. Instead of using procedural code, a classic expert system relies on 69 if-then-else logic to function. Instead of if-then-else rules, machine learning algorithms are utilized in 70 the workplace to create intelligent systems. The remainder paper is structured with: Section 2 71 discussed earlier studies as well as state-of-art methods and strategies. Furthermore, pre-processing 72 of data is discussed in Section 3, including dataset feasibility summary, feature visualizations, and 73 attribute extraction. Section 4 digs into categorization models and the underlying principles of each 74 method. Section 5 shows the efficiency measures, methods, and experimental setup. It also shows 75 how the experiment was carried out and the results acquired. Section 6 finishes with a recap of 76 previous work and many ideas for future improvements. 

# 77 **LITERATURE REVIEW** 

78 Simple lifestyle changes along with early care significantly increase the prognosis of heart attacks. 79 The multi-factorial aspect of multiple contributory risk factors such as diabetes, high blood pressure, 80 high cholesterol, and others makes it impossible to distinguish high-risk patients. Here is where data 81 processing and deep learning come in handy. By analysing thousands of clinical reports and other 82 medical data, machine learning algorithms can identify trends associated with diseases and health 83 conditions (Banerjee et al., 2022; Maaliw, Alon, Lagman, Garcia, Abante, et al., 2022; Özbay 84 Karakuş & Er, 2022). 

85 Several studies using various classifiers and function selection methods are performed on medical 86 data sets. For instance, in the computer science literature, various machine learning techniques have 87 been applied to cardiac arrest prediction (Chae et al., 2022; Dissanayake & Md Johar, 2021). For the 88 best analysis of cardiac disorders, Jindal et al. (2021) used a machine learning technique and 4% 89 KNN (87% accuracy) is considerably more efficient than SVP, Decision tree, and linear regression 90 algorithms. It also employs the usage of a confusion matrix to compare the accuracy of each method. 91 Meanwhile, Junaid and Kumar (2020) suggested a hybrid algorithm-based prediction of heart disease 92 in their research study. The sensitivity and specificity of this Hybrid are 91.47% and 82.11%, 93 respectively. Their study provides a novel route for experiments by demonstrating how these 94 principles might be used in smart devices and data mining to enhance cardiac arrest detection and 95 treatment. Depending on the circumstance, different methods performed better, whether cross96 validation, grid search, calibration, or feature selection were utilized or not. Any algorithm has the 97 inherent ability to outperform other algorithms. The average accuracy of the highest efficiency 98 without improving NB percentage is 83.6%, which is a high percentage of RF's 81.4% (Khourdifi & 99 Bahaj, 2018). On the other hand, Karthikeyan et al. (2021) found that ANN has a higher percentage 100 accuracy (85%) after comparing ANN, Logistic Regression, Decision Tree, Random Forest, and 101 Support Vector Machine algorithms to diagnose heart diseases in patients. Once a much larger 102 dataset is available, ANN may be employed. The number of hidden layers and epochs might also be 103 raised, which would improve the neural network's throughput. 

100 

101 

102 

103 

Another approach uses RNN and GRU to make the system more accurate and effective to predict silent heart attacks and inform the user at the earliest possible. This system increased the heart attack prediction accuracy to 92% and has proved to be an excellent source in predicting silent heart attacks (Kishore et al., 2018). The enhanced medical characteristics in the Dataset boost accuracy. Logistic 

104 

105 

106 

107 

108 Regression, Random Forest Classifier, and KNN are the methods utilized to create the provided 109 model. The accuracy was 87.5%, which is higher than the prior models' accuracy of 85%. As a 110 result, as the number of qualities rises, so does the accuracy (Jindal et al., 2021). In this study, two 111 additional input variables, obesity, and smoking produce more accurate outcomes. Decision trees, 112 Naive Bayes, and Neural Networks were used as data mining classification approaches. According to 113 the data, Neural Networks produce more accurate results than Decision Trees and Naive Bayes 114 (Priya et al., 2022). Meanwhile, Sultana et al. (2016) addresses the issue of the prediction of heart 115 disease according to some input attributes. Accuracy does not always give an accurate measure of the 116 performance of a classifier, as FPR is a dangerous error. For this reason, the classifier performance is 117 usually expressed using the Receiver Operating Characteristics (ROC) curve (that is, a plot of TPR 118 vs FPR). The performance of the Bayes Net classifier is the best, and that of the J48 classifier is the 119 worst. The performances of SMO, MLP, and KStar are also good. In another study, Rajendran and 120 Karthi (2022) used three classifiers (ID3, CART, and DT) to diagnose patients with heart diseases. 121 Observation shows that CART performance is having more accuracy when compared with the other 122 two classification methods. The best algorithm based on the patient’s data is CART Classification 123 with an accuracy of 83.49%, and the total time taken to build the model is at 0.23 seconds. CART 124 classifier has the lowest average error at 0.3 compared to others. These results suggest that the CART 125 classifier can significantly improve the conventional classification methods used in the study. 

126 In another study, Takci (2018) conducted experiments with and without feature selection to measure 127 the feature selection effect for heart attack prediction. Without feature selection, the best result, 128 based on model accuracy, gave many classifiers. Eight classifiers gave an accuracy of around 80%. 129 BLR and naïve Bayes gave the best result in terms of processing time. The same algorithms also 130 gave the best results according to model accuracy and processing time. Then the experiments were 131 repeated by selecting features. With feature selection, even though not the case for all algorithms, 132 some of them improved both processing time and model accuracy. The highest accuracy value was 133 82.59% without feature selection and it was improved to 84.81% with feature selection. SVM-linear 134 and naïve Bayes gave model accuracy of 84.81%. On the other hand, the results in the field of data 135 classification were obtained with the Naive Bayes algorithm, Decision list algorithm, and KNN 136 algorithm, and overall, the performance made known the Naive Bayes Algorithm when tested on 137 heart disease datasets (Rajkumar & Reena, 2010). Accordingly, it was found that the Naive Bayes 138 algorithm was the best compact time for processing datasets and showed better performance in 139 accuracy prediction. The time taken to run the data for the result is fast when compared to other 140 algorithms. It shows the enhanced performance according to its attribute. Attributes are fully 141 classified by this algorithm, and it gives 52.33% of accurate results. 

142 In a study conducted by Chitra and Seenivasagam (2013), the sensitivity achieved for the FCM 143 classifier is 91.53 with an average false positive of 0.9 per 30 records. The achieved accuracy is 144 92%, which is better than the performance of the neural network-based classifier and K-means 145 clustering algorithm. The increase in the performance of the FCM clustering algorithm is because the 146 weights of data attributes are set to adjust original samples to the uniform distribution, which could 147 be suitable for the character of FCM calculation to improve the accuracy. The results of the 148 classification experiment, performed over a data set obtained from 270 patients, show that the 149 classifier has achieved better accuracy than most of the existing algorithms. Meanwhile, Methaila et 150 al. (2014) focuses on using different algorithms and combinations of several target attributes for 

- 151 effective heart attack prediction using data mining. Decision Tree outperforms with 99.62% accuracy 

- 152 by using 15 attributes. Also, the accuracy of the Decision Tree and Bayesian Classification further 

- 153 improves after applying a genetic algorithm to reduce the actual data size to get the optimal subset of 

- 154 attributes sufficient for heart disease prediction. In principle, the BN learning algorithms can 

- 155 discover the mediated correlation since they test pairwise independence and conditional 

- 156 independence given values of other variables. Bayesian networks are a tool of choice for reasoning in 

- 157 uncertainty, with incomplete data. However, often, Bayesian network structural learning only deals 

- 158 with complete data. This study has proposed an adaptation of the learning process of the Chow–Liu, 

- 159 and TAN from incomplete and imbalanced datasets. These methods have been successfully tested on 

- 160 the dataset. It is seen that the TANI algorithm is a single winner with D.Bin (Salman, 2019). Finally, 

- 161 Dangare and Apte (2012) presented a Heart disease prediction system (HDPS) using data mining and 

- 162 artificial neural network (ANN) techniques. From the ANN, a multilayer perceptron neural network 

- 163 along with a Backpropagation algorithm was used to develop the system. The experimental result 

- 164 

- 165 

- shows that by using neural networks the system predicts heart disease with nearly 100% accuracy. 

**Table 1. Sample Content from the Dataset Used for Validation** 

|**age**|**sex**|**cp**|**trtbps**|**chol**|**fbs**|**restecg**|**thalachh**|**exng**|**oldpeak**|**slp**|**caa**|**thall**|**output**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|63|1|3|145|233|1|0|150|0|2.3|0|0|1|1|
|37|1|2|130|250|0|1|187|0|3.5|0|0|2|1|
|41|0|1|130|204|0|0|172|0|1.4|2|0|2|1|
|56|1|1|120|236|0|1|178|0|0.8|2|0|2|1|
|57|0|0|120|354|0|1|163|1|0.6|2|0|2|1|
|57|1|0|140|192|0|1|148|0|0.4|1|0|1|1|
|56|0|1|140|294|0|0|153|0|1.3|1|0|2|1|
|44|1|1|120|263|0|1|173|0|0|2|0|3|1|
|52|1|2|172|199|1|1|162|0|0.5|2|0|3|1|
|57|1|2|150|168|0|1|174|0|1.6|2|0|2|1|



166 

Note: The descriptions of the attributes are presented in Table 2. 

## 167 **Table 2. Description of the Attributes in the Dataset** 

|**Name**|**Type**|**Description**|
|---|---|---|
|age|Continuous|Age in years|
|sex|Discrete|0 = female, 1 = male|
|cp|Discrete|Chest Pain Type: 1 = typical angina, 2 = atypical angina, 3 = non-angina pain, 4 =<br>asymptom|
|trtbps|Continuous|Resting Blood Pressure (in mm Hg)|
|chol|Continuous|Serum Cholesterol (in mg/dl)|
|fbs|Discrete|Fasting Blood Sugar > 120mg/dl: 1 = True, 0 = False|
|restecg|Discrete|Rest Electrocardiograph: 0 = normal, 1 = having ST-T wave abnormality, 2 = left<br>ventricular hypertrophy|
|thalachh|Continuous|Maximum Heart rate achieved|
|exng|Discrete|Exercised induced angina: 0 = no, 1 = yes|



|oldpeak|Continuous|Depression induced by exercise relative to rest|
|---|---|---|
|slp|Discrete|The slope of the peak exercise segment: 1 = up sloping, 2 = flat, 3 = down sloping|
|caa|Continuous|The number of Major Vessels coloured by fluoroscopy ranged between 0 and 3.|
|thal|Discrete|Thalassemia defect type: 3 = normal, 6 = fixed defect, 7 = reversible defect|



168 

# 169 **SCENARIOS AND DATA DESCRIPTIONS** 

- 170 The Heart Disease dataset is utilized to conduct the validation of the proposed model. This Dataset 

- 171 was initially cited by the University of California, Irvine Repository (Mamun et al., 2022). This 

- 172 dataset comprises numerous attributes and a series of 303 entries collected from various heart 

- 173 patients’ data as depicted in Table 1 and Table 2. The corresponding graphical visualizations are 

- 174 stated in Figure 2 and Figure 3. Based on the attribute selections, the quantitative measure among 

- 175 Attribute-to-Attribute Co-variance is depicted in Figure 4a and the covariance heatmap is visualized 

- 176 in Figure 4b. 

- 177 



- 178 **Figure 2. Distribution Plots for the Numeric Attributes of the Dataset: (a) Age, (b) Resting Blood Pressure, (c)** 179 **Cholesterol, (d) ST Depression, (e) Major Blood Vessels, and (f) Max HR Achieved.** 

180 



181 **Figure 3. Histogram for the Categorical Attributes of the Dataset: (a) Sex, (b) Chest Pain Type, (c) Fasting Blood** 182 **Sugar, (d) Resting ECG, (e) Exercise Induced Angina, (f) ST Slope, (g) Thalassemia, and (h) Output Distribution.** 

183 



184 **Figure 4. (a) Attribute-to-Attribute Co-variance Size Scatterplot and (b) Co-variance Heatmap.** 

## 185 **Attribute Selection** 

186 The key features and the need to adjust the data were determined using Principal Component 187 Analysis (PCA). 





189 

190 



191 **Figure 5. (a) PCA scatterplot for major two components showing glaring distinctive clusters and (b) PCA** 

192 **cumulative variance distribution for the 13 attributes.** 

- 193 The feasibility of the dataset, as presented in 

194 



- 195 Figure **_5_** a, is checked via a PCA cluster graph, which justifies the purpose of the dependency of 

- 196 independent constraints on the dependent entity. On the other hand, the graph in 



197 

- 198 Figure **_5_** b suggests the subtle variance, which is observed using every attribute, making it necessary 199 to include every dependent quantity to predict the output and thus rules out the need for alteration of 200 the Dataset. 

## 201 **Pre-Processing** 

- 202 The given Dataset is unbalanced and can lead to an imbalance of classes in training and exacerbate 203 the accuracy and biased models, which justifies the reason for balancing the classes done by 204 resampling the minority upscale it to have the same frequency as of majority class. Furthermore, the 205 normalization of the numeric data is applied for simplifying the inputs, which facilitates the 

- 206 processing of training and testing of the Dataset. Class Balancing using resampling is done. Initially, 207 the Dataset comprised 165 entries with ‘0’ as output and 138 entries with ‘1’ as output. Furthermore, 208 it was upscaled to 330 entries with 165 from each of the classes to optimize the results. 

## 209 **Standardization** 

- 210 The process of tuning and rescaling characteristics so that the resultant attributes have a mean of 0 211 and a standard deviation of 1 is known as data standardization. It secures that the data is consistent. 212 The following formula is used to calculate the standardized score. 

- 213 

(𝑥 − 𝑥) 𝑠𝑡𝑑𝑖𝑧𝑒𝑑𝑉𝑎𝑙𝑢𝑒 = 𝜎 

- 214 

- 215 Here, 𝑥 is denoted as average, while 𝜎 depicts the standard deviation of the characteristic rows. 

- 216 

## **Train-Test Split** 

- 217 The Dataset is split into two sections: a training dataset that the model uses to fit to the Dataset, and a 218 testing dataset that confirms and supports the trained model's validity. The dataset split in this study 219 is 1:3, which implies that the training dataset makes up 75% of the whole dataset and the testing 220 dataset makes up 25% of the total dataset. 

- 221 

# **MODELS AND METHODOLOGY INVESTIGATIONS** 

## 222 **Random Forest** 

223 Decision Forests or Random Forests is an ensemble learning approach for regression as well as 224 classification problems which involves training many individual decision tree 

- 225 The Gini of each branch on a node, which specifies the more probable branches, is calculated using 226 the class and probability formula. Here 𝑝𝑎 denotes the occurrence, while _C_ depicts the number of 227 categories. 

228 

𝐶 𝐺𝑖𝑛𝑖 = 1 − ∑(𝑝𝑎)<sup>2</sup> 𝑎=1 

229 

240 

241 

242 

243 244 245 

246 

247 The k-nearest neighbors (KNN) algorithm is a simple, easy-to-use supervised machine learning 248 algorithm for classification and regression problems. The KNN algorithm assumes that identical 249 objects happen nearby. In other terms, related entities are close together (Desai et al., 2021). 

250 Algorithm for KNN: 

251 252 253 254 255 256 257 258 259 260 

- 230 Entropy is used to determine whether a node can branch depending on the likelihood of a specific 231 outcome. Because it is derived using a logarithmic equation (Singh et al., 2017), it is more 232 mathematically demanding than the Gini index. 

233 



- 234 **Supervised Learning Models** 

235 Supervised learning is a category of machine learning that uses named datasets to train algorithms to 236 identify data or predict outcomes correctly. When input data is fed into the algorithm, it changes its 237 weights using a reinforcement learning mechanism, ensuring that the model is adequately equipped. 

238 There are 6 Supervised Learning Models which are utilized to compare and analyze the behavior and 239 adaptability to the given Dataset. 

- K- Nearest Neighbours 

- Long Short-Term Memory (LSTM) 

- Artificial Neural Network (ANN) 

- Naïve Bayes 

- Logistic Regression 

- Support Vector Machine 

## _K-Nearest Neighbours_ 

|Step 1:|Loading the input data.|
|---|---|
|Step 2:|Set K to the number of neighbors|
|Step 3:|For each data example, calculate the distance from the data between the question<br>example and the present example. To an ordered set, add the distance and the<br>example’s index.|
|Step 4:|Sort the organized set of distances and indices by distance from smallest to greatest.|
|Step 5:|Choose the first K entries from the sorted collection.|
|Step 6:|Obtain the labels of the chosen K entries.|
|Step 7:|If there is regression, return the average of the K symbols.|
|Step 8:|If classification was performed, return the mode of the K marks.|



## 261 

## _Support Vector Machine_ 

- 262 The Support Vector Machine (SVM) is a basic algorithm that any machine learning professional 263 should have in his or her arsenal as the support vector machine. Many people choose support vector 264 machines because they achieve substantial precision by using fewer computing resources. 

- 265 Hyperplanes are judgment boundaries that aid in the classification of data points. Different groups 

- 266 may be assigned to data points that land on either side of the hyperplane. Furthermore, the size of the 267 hyperplane is determined by the number of functions. Support vectors are the data points that lie 

274 275 276 277 278 

- 268 closer to the hyperplane that controls the direction and orientation of the hyperplane. Using these 

- 269 help vectors, we optimize the classifier’s margin. The location of the hyperplane will change if the 270 support vectors are removed. These are the points that will assist us in developing our SVM (Amami 271 et al., 2015). 

272 

For a given Dataset 𝑆= {(𝑥1, 𝑡1), (𝑥2, 𝑡2), … , (𝑥𝑚, 𝑡𝑚) | (𝑥𝑘, 𝑡𝑘) ∈ 𝑅<sup>𝑛</sup> ∗{+1, −1}} 

- 273 Algorithm for SVM: 

Step 1: Create the template matrix. Step 2: Implement the Kernel Function and customize the parameters of Kernel Function and the value of C. 

Step 3: Run the training Algorithm to obtain the value of α Step 4: The value of α and support vectors are used to classify previously unseen data. 

279 

280 

Mathematical Formula: 



281 _Logistic Regression_ 

282 Logistic regression is a mathematical model which assesses if an independent variable influences a 283 binary dependent variable. It implies that for a given input, there are only two possible outcomes. 

284 A sigmoid function is a common model. The sigmoid function, also known as a squashing function, 285 confines outputs to the range between 0 and 1. 

286 



287 There are two variables, b0, and b1, in the function above. These are referred to as the weights or 288 coefficient values. The bias, or intercept, is represented by b0, and b1 represents the coefficient. The 289 actual data set is used to study and train these weights. This calculation would yield a percentage or 290 expectation that will be mapped over discrete groups. The judgment boundary is the specified 291 distinction between two groups (Manogaran & Lopez, 2018). 

292 _Naïve Bayes_ 

293 Naïve Bayes classifiers are a subset of basic “probabilistic classifiers” based on applying Bayes’ 294 theorem to features with strict independence assumptions. They are among the most basic Bayesian 295 network models, but they can reach higher levels of accuracy when combined with kernel density 296 estimation. 

297 To our Dataset, we can now apply Bayes’ theorem as follows: 

298 



299 Where y is a class variable and X is an n-dimensional dependent function vector (Dulhare, 2018). 

300 _X = (x1, x2, x3,…, xn)_ 

- 301 _ANN Sequential Model_ 

325 

327 

- 302 Keras is an open-source deep learning platform for Python with a simple structure, which provides a 303 clean and straightforward way to construct TensorFlow-based deep learning models. Keras Layers 

- 304 are the basic building blocks of Neural Networks, where a layer is made up of tensor-input and 

- 305 tensor-output calculation functions and a state in TensorFlow variables. Tensors are 306 multidimensional arrays of a standardized data type. Tensors are both immutable. 

- 307 Keras layer needs an input shape and an initializer to set the weight for each to understand the input 308 data. The constraints parameter limits and defines the spectrum of input data weighting to be 309 generated. The regularise optimizes the layer and the model by adding penalties to the weights 310 dynamically during the optimization process. The input Shape must be given to the network’s first 311 layer. 

- 312 Keras Layers are the rudimentary components of Neural Networks. A tensor-input and tensor-output 313 computation function and a state stored in TensorFlow variables make up a layer. Tensors have 314 several dimensions. 

- 315 The Sequential model can be described as a network of Dense Layers utilized to train the deep 316 learning framework. The proposed model uses Sequential Model, which enables a model fitted with 317 multiple layers, where each layer has a weight that supplements the layer that follows it (Nalavade et 318 al., 2014). 

- 319 A total of 9 layers are applied, out of which one layer is for input, seven hidden layers, and the final 320 layer is for the output purpose. All the layers except the last layer implement the Swish Activation, a 321 soft curve and non-monotonic function and unbounded, which is a desirable attribute for an 322 activation function. It can avoid possible problems when the gradients are nearly zero and thus 323 outperform the traditional ReLU activation function. The swish function applies the following 324 mathematical formula: 

𝑌= 𝑋∗ σ(X) 

326 While the final layer uses Sigmoid Activation to predict the binary class. 

_LSTM Sequential Model_ 

- 328 The Long Short-Term Memory (LSTM) Network is an innovative RNN (sequential network) that 329 allows information to be stored indefinitely. It will deal with the vanishing gradient problem that 330 RNN has. For persistent memory, a recurrent neural network, also known as RNN, is used. RNNs 331 recall facts from the past and apply them to the new input. Because of the vanishing gradient, RNNs 332 are unable to remember long-term dependencies. Long-term dependence issues are expressly avoided 333 for LSTMs (Islam et al., 2019). A total of three layers were associated with the Sequential Model to 334 implement the LSTM functionalities, with the usage of ‘Swish’ activation. 

- 335 **Bagging Models** 

336 Bagging or bootstrap aggregating is a machine learning ensemble meta-algorithm for improving the 337 consistency and accuracy of machine learning algorithms used for statistical classification and 338 regression. It also helps to prevent overfitting by reducing variance. While it is most often associated 339 with decision tree systems, it can be applied to any design. Bootstrap Aggregation, or bagging for 340 short, is a straightforward and highly effective ensemble process. 

341 

342 

343 344 345 

346 

353 

360 • Builds several bootstraps = False by example, so samples are divided into arbitrary divisions 361 between a random subset of functions picked at each node without any substitute nodes. 362 • Nodes are divided based on arbitrary divisions between a random subset of selected features 363 of each node. 

364 

370 371 372 373 374 

The following models are ensembled with a bagging approach: 

- Voting Classifier 

- Extra-Trees 

- Random Forest (Proposed Model) 

- Decision Tree 

## _Voting Classifier_ 

347 The Voting classifier is not a classifier in and of itself but rather a wrapper for a set of different ones 348 trained and evaluated in parallel to leverage the unique characteristics of each algorithm. Classifier, 349 which involves voting of the other models to provide a unified class. Dataset can be trained using 350 various algorithms and then use an ensemble to predict the final result (Latha & Jeeva, 2019). The 351 Voting classifier has been utilized by implementing a selection of five KNN models with different 352 neighbour values: KNN 1, KNN 3, KNN 5, KNN 7, and KNN 9. 

## _Extra Trees_ 

354 Extra Trees Classifier is an ensemble learning system focused on decision trees. Extra Trees 355 Classifier, like Random Forest, randomizes some decisions and subsets of data to reduce over356 learning and overfitting from data. It uses the same Formula as the Decision Tree. Extra Trees is like 357 Random Forest. It creates several trees and divides nodes using random subsets of elements, but with 358 two main differences: it does not bootstrap observations (samples without replacement). The nodes 359 are divided into random splits rather than better splits. So, In all Extra Trees: 

- Builds several bootstraps = False by example, so samples are divided into arbitrary divisions between a random subset of functions picked at each node without any substitute nodes. 

## _Decision Tree_ 

365 A decision tree is a decision-making method that employs a tree-like model of decisions and their 366 potential outcomes, such as chance case outcomes, resource costs, and utility. It is one method of 367 displaying an algorithm that consists solely of conditional control statements (Reddy et al., 2016). 

368 

- 369 

Step 1: It starts with the initial root node, S. Step 2: The algorithm iterates over the very unused attribute of the set S and measures the Entropy (H) and Information gain (IG) of this attribute for each iteration. Step 3: It then chooses the attribute with the lowest Entropy or highest Information gain. Step 4: The chosen attribute then subdivides the set S to yield a subset of the results. 

Step 5: The algorithm proceeds to recurse on each subset, considering only attributes that have never been picked before. 

375 





377 

384 

385 

386 

387 388 389 

390 

391 

397 

398 Sigma Residual is the sum of residuals of that specific leaf. The denominator suggests the sum of the 399 prior probability of success and likelihood of failure (Bahad & Saxena, 2020). 

404 

405 

406 

407 

408 409 

## **Boosting Models** 

- 378 Boosting is an ensemble simulation technique that aims to create a robust classifier out of many 379 weak ones. It is achieved by combining weak models to build a model. Firstly, a model is created 380 using the training data. The second model is then developed, which attempts to correct the errors in 381 the first model. This process is repeated until either the entire training data set is correctly estimated, 382 or the total number of models has been applied. In all, six boosting models have been implemented 383 on the Dataset to contrast the functionalities and results, namely: 

   - Gradient Boosting 

   - XGB 

   - CatBoost 

   - Gaussian Process 

   - LGBM 

   - AdaBoost 

   - SGD 

## _Gradient Boosting_ 

392 Gradient boosting classifiers are a set of machine learning algorithms that combine many poor 393 learning models to construct a robust predictive model. When doing gradient boosting, decision trees 394 are commonly used. The concept behind “gradient boosting” is to take a weak hypothesis or learning 395 algorithm and make a set of tweaks to increase the hypothesis’s power. The idea of Probability 396 Approximately Correct (PAC) Learning underpins this method of hypothesis boosting (PAC). 



- 400 _XGB_ 

401 XGB is an open-source software library that offers a regularising gradient boosting mechanism to 402 provide Scalable, Portable, and Distributed Gradient Boosting. The following features of XGBoost 403 distinguish it from other gradient boosting algorithms: 

   - Trees are penalized cleverly. 

   - A proportional reduction in the size of leaf nodes 

   - Boosting Newton 

   - An additional parameter for randomization 

   - Out-of-core computing and implementation on independent, distributed systems 

   - A Feature collection is made automatically. 

- 410 XGBoost builds trees using the loss function, which minimizes the following value: 







418 

420 

423 424 425 426 427 

428 

429 CountInClass is the number of times the mark value for objects with the current categorical function 430 value was equal to “1.”. The numerator’s preliminary value is called prior. The beginning parameters 431 decide this. The cumulative number of items (up to the current one) with a categorical function 432 attribute matching the current one is called TotalCount. It can be expressed mathematically using the 433 following equation: 

434 

435 

- 413 The goal is to find an optimal output value for the leaf to reduce the overall equation to a minimum. 414 Since that starts with a value of y0, the following prediction is always equal to the i-1<sup>st</sup> forecast plus 415 the output value from the i<sup>th</sup> tree. As a result, we replace the first element, as seen below. 



416 

- 417 The Final Similarity score is calculated using the below formula [31]. 

- 419 CatBoost 

421 

422 

Yandex created CatBoost, an open-source software library. It offers a gradient boosting mechanism that, in contrast to the classical algorithm, aims to solve categorical features using a permutationdriven alternative. It provides out-of-the-box solid support for the more descriptive data formats that surround many market challenges. It produces state-of-the-art outcomes without the intensive data training that other machine learning approaches need. The procedure is as follows: 

1. Randomly permute the set of input observations. Many random permutations are generated. 

2. Converting a floating-point or unit mark value to an integer 

3. The following function is used to convert all categorical attribute values to integer values: 

countInClass + prior avg_target = 𝑡𝑜𝑡𝑎𝑙𝐶𝑜𝑢𝑛𝑡 + 1 

Let σ = (σ1, …, σn) be the permutation, then 𝑥σ𝑝,𝑘 is substituted by 



## 436 

## _Gaussian Process_ 

437 The Gaussian Processes Classifier is a machine learning classification algorithm. Gaussian Processes 438 generalizes the Gaussian probability distribution used to build advanced non-parametric machine 439 learning algorithms for classification and regression. The distribution of random variables is 440 summarised by Gaussian probability distribution functions, while the properties of the functions, 441 such as the parameters of the functions, are summarised by Gaussian processes. As a result, Gaussian 442 processes can be thought of as one degree of complexity or indirection above Gaussian functions. 443 The concept behind Gaussian Process Regression is that we conclude that a series of observed values 444 FN at some points XN correspond to the realization of a multivariate Gaussian Process with a prior 445 distribution: 



446 

453 454 455 

464 

473 

447 The coefficients of KN are expressed in terms of a correlation function (or kernel) Kmn =K(xm,xn). 448 The kernel’s hyper-parameters are calibrated using the maximum probability theorem. KN is chosen 449 to represent a function’s prior expectation, and thus the kernel’s selection would have a substantial 450 effect on the regression’s correctness (Manogaran & Lopez, 2018) **.** 

## 451 _LGBM_ 

452 

456 

457 

458 

459 

460 

461 

462 

463 

LightGBM (Light Gradient Boosting Machine) is a free and open source distributed gradient boosting platform for machine learning created by Microsoft. It is used for ranking, sorting, and other machine learning tasks built on decision tree algorithms. Like CatBoost, it can accommodate categorical attributes by using function names as feedback. It is also simpler than one-hot coding and does not translate to one-hot coding. To determine the split worth of categorical functions, LGBM employs a unique algorithm. It employs two innovative techniques: Gradient-based One Side Sampling and Exclusive Feature Bundling (EFB), which address the drawbacks of the histogrambased algorithm found in most GBDT (Gradient Boosting Decision Tree) frameworks. The characteristics of the LightGBM Algorithm are formed by the two techniques of GOSS and EFB mentioned below. They work together to make the model run smoothly and give it an advantage over other GBDT systems. In the computation of knowledge gain, different data instances play different roles. The knowledge advantage would be more significant for samples with higher gradients. 

## _AdaBoost_ 

465 Adaptive Boosting is a predictive grouping meta-algorithm that can increase efficiency when 466 combined with various learning algorithms. The results of the other learning algorithms are compiled 467 into a weighted sum that represents the boosted classifier’s final output. AdaBoost is adaptive in that 468 it tweaks subsequent vulnerable learners in Favor of instances misclassified by prior classifiers. It 469 could be less prone to the overfitting dilemma than other learning algorithms in certain situations. 470 Individual learners will be poor, but if their success is marginally higher than random guessing, the 471 final model may converge to a good learner. AdaBoost is a specific teaching system for boosted 472 classifiers. 



474 Every ft is a poor learner that takes an object x as input and returns a value that indicates the object’s 475 class. The sign of the weak learner performance, for example, identifies the expected object type in 476 the two-class puzzle, while the absolute value indicates the belief in that classification. Similarly, if 477 the sample belongs to a positive class, the T<sup>th</sup> classifier is positive; otherwise, it is negative (Fitriyani 478 et al., 2020). For each example in the training set, each weak learner generates an output hypothesis, 479 h(xi). A bad learner is chosen and given a coefficient αt at each iteration _t_ such that the sum training 480 error _Et_ of the resulting t-stage boost classifier is minimized. 

481 



- 482 Ft-1(x) is the boosted classifier built up to the previous stage of the study, E(F) is any error feature, 483 and ft(x)= αt h(x) is the slow learner under consideration for inclusion in the final classifier. Every 484 iteration of the training method assigns a weight wi,t to each sample in the training set equal to the 485 current error E(Ft-1(xi)) on that sample. These weights can be used to inform the teaching of the slow 

498 

517 same. 518 • **True Negative** : when the person has not suffered from heart disease and the model predicts 519 the same. 520 • **False Negative** : when the person has suffered from heart disease, and the model predicts the 521 opposite. 522 • **False Positive** : when the person has not suffered from heart disease, and the model predicts 523 the opposite. 

486 learner; for example, decision trees that support separating sets of samples with high weights can be 487 grown (Bahad & Saxena, 2020). 

488 _SGD_ 

489 The term “stochastic” refers to a system or process that is subject to random chance. As a result, 490 instead of selecting the entire data set for each iteration in Stochastic Gradient Descent, a few 491 samples are chosen at random. The term “batch” is used in Gradient Descent to refer to the total 492 number of samples from a dataset used to calculate the gradient for each iteration. The batch is taken 493 to be the entire Dataset in traditional Gradient Descent optimization, such as Batch Gradient Descent. 494 While using the whole Dataset is extremely useful for obtaining the minima with less noise, 495 randomly, the problem arises when our datasets become large. This problem is solved using 496 Stochastic Gradient Descent. SGD performs each iteration with a single sample, i.e., a batch size of 497 one. The sample is randomly shuffled and chosen for iteration. 

499 



500 The gradient of the cost function of a single example is determined at each iteration rather than the 501 sum of the slopes of all the samples. Since only one sample from the Dataset is selected at random 502 for each iteration in SGD, the direction taken by the algorithm to reach the minima is generally 503 noisier than in a traditional Gradient Descent algorithm. However, that does not matter because the 504 path taken by the algorithm is irrelevant as long as we meet the minima in a slightly shorter amount 505 of time. 

# 506 **EXPERIMENTAL RESULTS AND DISCUSSIONS** 

507 The Train-Test split approach splits the data arrays into two subsets, one for testing and the other for 508 preparation. The Dataset has been divided into two parts: a 25% testing set and a 75% training set. 509 The Random Forest algorithm has been applied to predict the dataset with utmost accuracy. The 510 proposed random forest framework is measured in terms of accuracy for metrics, and scores are 511 compared based on sensitivity, specificity, accuracy, precision, and F1-score. All the scores are 512 derived from the confusion matrix. For the binary classification, the model predicts the output ‘1’, 513 when the person is likely to suffer from heart disease, and ‘0’ if the person is not expected to suffer 514 from heart disease. For every classified output, it can be categorized into four categories as presented 515 in Figure 6: 

516 

- **True Positive** : when the person has suffered from heart disease, and the model predicts the same. 

- **True Negative** : when the person has not suffered from heart disease and the model predicts the same. 

- **False Negative** : when the person has suffered from heart disease, and the model predicts the opposite. 

524 

525 **Figure 6. Confusion Matrix Category Distribution** 



526 

527 

528 **Table 3. Parameters for the Bagging Models** 

|**Model Type**|**n_estimators**|**Voting**|**max_features**|
|---|---|---|---|
|Voting Classifier|KNN1, KNN3, KNN5,<br>KNN7, KNN9|Soft|-|
|ExtraTrees Classifier|100|-|sqrt|
|Random Forest|300|-|sqrt|
|Decision Tree|-|-|sqrt|



529 

530 **Table 4. Parameters for the Supervised Learning Models** 

|**Model Type**|**Parameters**|
|---|---|
|SVM|C: [0.1, 1, 10, 100, 1000], Gamma = [1, 0.1, 0.01, 0.001, 0.0001]|
|KNN|K value: 49|
|Logistic Regression|-|
|Naive Bayes|-|
|ANN|Dense Layers: 12/32/128/256/256/64/16/8/1, Activation: Swish, Sigmoid for the last layer|



531 

|LSTM|LSTM Layers: 64/512/2048/128, Activation: Swish, Sigmoid for the last layer|
|---|---|



## 532 

**Table 5. Parameters for the Boosting Models** 

|**Model Type**|**Parameters**|
|---|---|
|SGD|max_iter = 1000, tol = 0.01|
|AdaBoost|n_estimators = 50, learning_rate = 1|
|LGBM|-|
|Gaussian Process|kernel = 1.0 * RBF (1.0)|
|CatBoost|n_estimators = 100|
|XGB|n_estimators = 10, objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,<br>max_depth = 5, alpha = 10|
|Gradient Boosting|-|



## 533 

## 534 

**Table 6. Analysis of Scores of Bagging Models** 

|**Model**<br>**Type**|**Recall**|**F1 Score**|**Precision**|**Specificity**|**Sensitivity**|**Testing**<br>**Accuracy**|**Training**<br>**Accuracy**|
|---|---|---|---|---|---|---|---|
|Decision<br>Tree|86|100|91|91|86|88.41|100|
|Random<br>Forest|92|100|100|100|92|95.65|100|
|Extra<br>Trees|94|100|94|100|89|94.2|100|
|Voting<br>Classifier|75|100|75|76|75|75.36|90.34|



## 535 

## 536 **Table 7. Analysis of Different Supervised Learning Models** 

|**Model**<br>**Type**|**Recall**|**F1 Score**|**Precision**|**Specificity**|**Sensitivity**|**Testing**<br>**Accuracy**|**Training**<br>**Accuracy**|
|---|---|---|---|---|---|---|---|
|SVM|83|100|83|88|78|82.61|67.63|
|KNN|78|100|90|91|78|84.06|63.77|
|Logistic<br>Regression|92|100|92|91|92|91.3|84.54|
|Naive<br>Bayes|88|100|88|91|86|88.41|82.61|
|ANN|92|100|97|97|92|94.2|100|
|LSTM|92|100|97|97|92|94.16|100|



537 

539 

## 538 **Table 8. Analysis of Scores of Boosting Models** 

|**Model**<br>**Type**|**Recall**|**F1 Score**|**Precision**|**Specificity**|**Sensitivity**|**Testing**<br>**Accuracy**|**Training**<br>**Accuracy**|
|---|---|---|---|---|---|---|---|
|SGD|80|100|80|94|67|79.71|68.6|
|AdaBoost|88|100|88|88|89|88.41|95.65|
|LGBM|93|100|93|97|89|92.75|100|
|Gaussian<br>Process|91|100|91|97|86|91.3|87.92|
|CatBoost|93|100|93|97|89|92.75|92.27|
|XGB|86|100|86|94|78|85.51|84.54|
|Gradient<br>Boosting|89|100|89|95|84|89.16|100|



- 540 The performance of the state-of-the-art machine learning approaches like Gradient Boosting 541 Classifier, SGD Classifier, Cat Boost Classifier, LGBM Classifier, XGB Classifier, AdaBoost 542 Classifier, Gaussian Process Classifier, LSTM RNN Model, Keras Sequential ANN Model, Naïve 543 Bayes, Logistic Regression, KNN, Support Vector Machine, Voting Classifier, Decision Tree, and 544 ExtraTrees Classifier is juxtaposed with the Random Forest model, also depicted in Tables 4 to 8. 545 According to the results, the Random Forest model produces optimal efficiency. Figures 7 to 9 depict 

- 546 the comparative performance assessments of the classifier models by considering all popular 

- 547 classification algorithms to identify the higher accuracy classifier model to predict heart disease 

- 548 automatically. When it comes to accuracy, we discovered that a less computationally intensive 

- 549 strategy like random forests outperforms a more complicated technique like Neural Networks. The 550 key reason for this is the tiny size of our Dataset, which can forecast the likelihood of cardiac arrest. 551 The experiments were run with and without feature selection to assess the impact of feature selection 552 on heart attack prediction. 

553 554 **Figure 7. Training and Testing Accuracy of (a) Bagging, (c) Supervised, and (e) Boosting Models and the Recall,** 555 **F1-Score, Precision, Specificity, and Sensitivity (b) Bagging, (d) Supervised, and (f) Boosting Models.** 

556 



557 **Figure 8. ROC Curve for Bagging Models: (a) Decision Tree, (b) Extratrees, (c) Random Forest, and (d) Voting Classifier** 

558 



559 **Figure 9. ROC Curve for (a) Boosting, (b) Bagging, and (c) Supervised Models and Compiled ROC Curves for Supervised and** 560 **Boosting Models.** 

# 561 **SOCIAL HEALTH EDUCATIONAL IMPLICATIONS** 

562 A visual analysis may accurately predict cardiac arrest, making it a potent educational tool for raising 563 public awareness of health issues. By predicting cardiac arrest earlier, preventative steps can be taken 564 to save lives. Promoting such health knowledge in society can dramatically lower the world mortality 565 rate. Cardiovascular disease is a term used to describe conditions affecting the heart and blood 566 arteries. Heart disease-related deaths have been rising quickly. The deadliest cause of death 567 worldwide is thought to be cardiovascular disease. The heart disease dataset from the UCI Machine 568 Repository is used in this study to provide an accurate diagnosis of heart illness. Using an 

582 583 

587 588 589 590 591 592 

593 

595 Amami, R., Ayed, D. B., & Ellouze, N. (2015). An Empirical Comparison of SVM and Some Supervised Learning 596 Algorithms for Vowel Recognition. _arXiv_ . https://doi.org/10.48550/arXiv.1507.06021 597 Amini, M., Zayeri, F., & Salehi, M. (2021). Trend Analysis of Cardiovascular Disease Mortality, Incidence, and 598 Mortality-to-incidence Ratio: Results From Global Burden of Disease Study 2017. _BMC Public Health_ , _21_ 599 1-12. https://doi.org/10.1186/s12889-021-10429-0https://doi.org/10.1186/s12889-021-10429-0-021-10429-0021-10429-0-10429-010429-0-00 600 Bae, S., Kim, S. R., Kim, M.-N., Shim, W. J., & Park, S.-M. (2021). Impact of Cardiovascular Disease and Risk Factors 601 on Fatal Outcomes in Patients With COVID-19 According to Age: A Systematic Review and Meta-analysis. 602 _Heart_ , _107_ (5), 373-380. https://doi.org/10.1136/heartjnl-2020-317901https://doi.org/10.1136/heartjnl-2020-317901-2020-3179012020-317901-317901317901 603 Bahad, P., & Saxena, P. (2020). Study of AdaBoost and Gradient Boosting Algorithms for Predictive Analytics. 604 _International Conference on Intelligent Computing and Smart Communication 2019_ , 235-244. 605 <u>https://doi.org/10.1007/978-981-15-0633-8_22</u> 606 Banerjee, P., Bhattacherjee, S., Dasgupta, K., & Sen, S. (2022). Performance Evaluation of Machine Learning Classifiers 607 for Sudden Cardiac Arrest Detection. _Journal of The Institution of Engineers (India): Series B_ . 608 <u>https://doi.org/10.1007/s40031-022-00830-7-022-00830-7022-00830-7-00830-700830-7-77</u> 609 Chae, M., Gil, H.-W., Cho, N.-J., & Lee, H. (2022). Machine Learning-Based Cardiac Arrest Prediction for Early 610 Warning System. _Mathematics_ , _10_ (12), 1-17. https://doi.org/10.3390/math10122049 611 Chitra, R., & Seenivasagam, V. (2013). Heart Attack Prediction System Using Fuzzy C Means Classifier. _IOSR Journal_ 612 _of Computer Engineering_ , _14_ (2), 23-31. https://doi.org/10.9790/0661-1422331https://doi.org/10.9790/0661-1422331-14223311422331 

569 optimization algorithm to diagnose heart disease can be beneficial in terms of improved sensitivity 570 and accuracy. Optimization is the process of selecting the optimal solution from all feasible solutions 571 to a given problem. Through the prediction of intelligent cardiovascular disease, the best outcomes 572 are produced by under-researched categorization algorithms when they are equipped with a gradient 573 descent optimization model. The need to create a model for accurately and efficiently diagnosing 574 heart disease has increased due to the rise in heart failure-related fatalities. The provision of 575 dependable facilities at reasonable prices is a major issue for healthcare institutions (healthcare 576 centres, hospital clinics). The key issues with the current models are accuracy, utility, and reliability. 577 The goal of the study is to identify the most efficient machine-learning method for making more 578 precise, sensitive, and accurate earlier predictions of heart disease so that curative actions can be 579 taken in advance to save human lives in society. 

## 580 

# **CONCLUSION AND FUTURE WORKS** 

581 

584 585 586 

Visual analysis is a powerful learning tool for promoting health awareness in society by earlier predicting cardiac arrest. Visual analysis is a powerful instructional tool for educating the public about health issues since it may properly predict cardiac arrest. Preventative measures to save lives can be implemented via earlier cardiac arrest prediction. A significant reduction in global death rates can be achieved by promoting such health information in society. This research uses the Random Forest approach to predict heart disease automatically. The experiment was conducted using the publicly available heart disease dataset. Before utilizing the input data to train and test the suggested model, it was necessary to undertake class balance through up-sampling minority classes and data standardization using the Z-score approach. Seventeen eclectic machine-learning techniques have been applied for comparing the alignment of the proposed model. The random forest produces the highest efficiency, according to empirical evidence. In the future, the suggested approach will be evaluated against a fresh heart disease dataset and used to forecast heart disease on a web-based platform to enable global accessibility through low-end computing devices. 

## 594 

# **REFERENCES** 

- Amini, M., Zayeri, F., & Salehi, M. (2021). Trend Analysis of Cardiovascular Disease Mortality, Incidence, and Mortality-to-incidence Ratio: Results From Global Burden of Disease Study 2017. _BMC Public Health_ , _21_ (1), 1-12. https://doi.org/10.1186/s12889-021-10429-0https://doi.org/10.1186/s12889-021-10429-0-021-10429-0021-10429-0-10429-010429-0-00 

- Bae, S., Kim, S. R., Kim, M.-N., Shim, W. J., & Park, S.-M. (2021). Impact of Cardiovascular Disease and Risk Factors on Fatal Outcomes in Patients With COVID-19 According to Age: A Systematic Review and Meta-analysis. _Heart_ , _107_ (5), 373-380. https://doi.org/10.1136/heartjnl-2020-317901https://doi.org/10.1136/heartjnl-2020-317901-2020-3179012020-317901-317901317901 

- Banerjee, P., Bhattacherjee, S., Dasgupta, K., & Sen, S. (2022). Performance Evaluation of Machine Learning Classifiers for Sudden Cardiac Arrest Detection. _Journal of The Institution of Engineers (India): Series B_ . <u>https://doi.org/10.1007/s40031-022-00830-7-022-00830-7022-00830-7-00830-700830-7-77</u> 

- Chitra, R., & Seenivasagam, V. (2013). Heart Attack Prediction System Using Fuzzy C Means Classifier. _IOSR Journal of Computer Engineering_ , _14_ (2), 23-31. https://doi.org/10.9790/0661-1422331https://doi.org/10.9790/0661-1422331-14223311422331 

613 Chudzińska, M., Wołowiec, Ł., Banach, J., Rogowicz, D., & Grześk, G. (2022). Alcohol and Cardiovascular 614 Diseases&mdash;Do the Consumption Pattern and Dose Make the Difference? _Journal of Cardiovascular_ 615 _Development and Disease_ , _9_ (10), 1-10. <u>https://doi.org/10.3390/jcdd9100317</u> 616 Dangare, C., & Apte, S. (2012). A Data Mining Approach for Prediction of Heart Disease Using Neural Networks. 617 _International Journal of Computer Engineering and Technology (IJCET),_ , _3_ (3), 30-40. 618 <u>https://www.researchgate.net/publication/254938414</u> 619 Desai, N. P., Wadhwani, A., Baluch, M. F., & Mishra, N. (2021). A Comparative Assessment Study on Machine 620 Learning Classifiers for Cardiac Arrest Diagnosis and Prediction. _2021 International Conference on Innovative_ 621 _Computing, Intelligent Communication and Smart Electrical Systems (ICSES)_ , 1-6. 622 <u>https://doi.org/10.1109/ICSES52305.2021.9633898</u> 623 Dissanayake, K., & Md Johar, M. G. (2021). Comparative Study on Heart Disease Prediction Using Feature Selection 624 Techniques on Classification Algorithms. _Applied Computational Intelligence and Soft Computing_ , _2021_ , 1-17. 625 <u>https://doi.org/10.1155/2021/5581806</u> 626 Dulhare, U. N. (2018). Prediction System for Heart Disease Using Naive Bayes and Particle Swarm Optimization. 627 _Biomedical Research_ , _29_ (12), 2646-2649. <u>https://doi.org/10.4066/biomedicalresearch.29-18-620</u> 628 Fitriyani, N. L., Syafrudin, M., Alfian, G., & Rhee, J. (2020). HDPM: An Effective Heart Disease Prediction Model for a 629 Clinical Decision Support System. _IEEE Access_ , _8_ , 133034-133050. 630 <u>https://doi.org/10.1109/ACCESS.2020.3010511</u> 631 Garcia, M. B., Ambat, S., & Adao, R. T. (2019). Tomayto, Tomahto: A Machine Learning Approach for Tomato 632 Ripening Stage Identification Using Pixel-Based Color Image Classification. _2019 IEEE 11th International_ 633 _Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control,_ 634 _Environment, and Management ( HNICEM )_ , 1-6. <u>https://doi.org/10.1109/HNICEM48295.2019.9072892</u> 635 Garcia, M. B., & Garcia, P. S. (2023). Intelligent Tutoring System as an Instructional Technology in Learning Basic 636 Nutrition Concepts: An Exploratory Sequential Mixed Methods Study. In M. B. Garcia, M. V. López-Cabrera, 637 & R. P. P. de Almeida (Eds.), _Instructional Technologies in Health Education and Allied Disciplines_ . IGI 638 Global. https://doi.org/10.4018/978-1-6684-7164-7 639 Islam, M. S., Umran, H. M., Umran, S. M., & Karim, M. (2019). Intelligent Healthcare Platform: Cardiovascular Disease 640 Risk Factors Prediction Using Attention Module Based LSTM. _2019 2nd International Conference on Artificial_ 641 _Intelligence and Big Data (ICAIBD)_ , 167-175. <u>https://doi.org/10.1109/ICAIBD.2019.8836998</u> 642 Javeed, A., Khan, S. U., Ali, L., Ali, S., Imrana, Y., & Rahman, A. (2022). Machine Learning-Based Automated 643 Diagnostic Systems Developed for Heart Failure Prediction Using Different Types of Data Modalities: A 644 Systematic Review and Future Directions. _Computational and Mathematical Methods in Medicine_ , _2022_ , 1-30. 645 <u>https://doi.org/10.1155/2022/9288452</u> 646 Jindal, H., Agrawal, S., Khera, R., Jain, R., & Nagrath, P. (2021). Heart Disease Prediction Using Machine Learning 647 Algorithms. _IOP Conference Series: Materials Science and Engineering_ , _1022_ (1), 1-10. 648 <u>https://doi.org/10.1088/1757-899X/1022/1/012072</u> 649 Junaid, M. J. A., & Kumar, R. (2020). Data Science and its Application in Heart Disease Prediction. _2020 International_ 650 _Conference on Intelligent Engineering and Management (ICIEM)_ , 396-400. 651 <u>https://doi.org/10.1109/ICIEM48762.2020.9160056</u> 652 Jurgens, C. Y., Lee, C. S., Aycock, D. M., Masterson Creber, R., Denfeld, Q. E., DeVon, H. A., Evers, L. R., Jung, M., 653 Pucciarelli, G., Streur, M. M., Konstam, M. A., & null, n. (2022). State of the Science: The Relevance of 654 Symptoms in Cardiovascular Disease and Research: A Scientific Statement From the American Heart 655 Association. _Circulation_ , _146_ (12), 173-184. https://doi.org/10.1161/CIR.0000000000001089 656 Karthikeyan, R., Vijendra Babu, D., Ekarthik, Suresh, R., Nalathambi, M., & Dinakaran, S. (2021). Cardiac Arrest 657 Prediction using Machine Learning Algorithms. _Journal of Physics: Conference Series_ , _1964_ (6), 1-9. 658 <u>https://doi.org/10.1088/1742-6596/1964/6/062076</u> 659 Khourdifi, Y., & Bahaj, M. (2018). Heart Disease Prediction and Classification Using Machine Learning Algorithms 660 Optimized by Particle Swarm Optimization and Ant Colony Optimization. _International Journal of Intelligent_ 661 _Engineering and Systems_ , _12_ (1), 242-252. <u>https://doi.org/10.22266/ijies2019.0228.24</u> 662 Kishore, A., Kumar, A., Singh, K., Punia, M., & Hambir, Y. (2018). Heart Attack Prediction Using Deep Learning. 663 _International Research Journal of Engineering and Technology_ , _5_ (4), 4420-4423. 664 <u>https://www.irjet.net/archives/V5/i4/IRJET-V5I4982.pdf</u> 665 Latha, C. B. C., & Jeeva, S. C. (2019). Improving the Accuracy of Prediction of Heart Disease Risk Based on Ensemble 666 Classification Techniques. _Informatics in Medicine Unlocked_ , _16_ , 1-9. 667 <u>https://doi.org/10.1016/j.imu.2019.100203</u> 668 Maaliw, R. R., Alon, A. S., Lagman, A. C., Garcia, M. B., Abante, M. V., Belleza, R. C., Tan, J. B., & Maaño, R. A. 669 (2022). Cataract Detection and Grading Using Ensemble Neural Networks and Transfer Learning. _2022 IEEE_ 670 _13th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON)_ , 74-81. 671 <u>https://doi.org/10.1109/IEMCON56893.2022.9946550</u> 

- Islam, M. S., Umran, H. M., Umran, S. M., & Karim, M. (2019). Intelligent Healthcare Platform: Cardiovascular Disease Risk Factors Prediction Using Attention Module Based LSTM. _2019 2nd International Conference on Artificial Intelligence and Big Data (ICAIBD)_ , 167-175. <u>https://doi.org/10.1109/ICAIBD.2019.8836998</u> 

- Javeed, A., Khan, S. U., Ali, L., Ali, S., Imrana, Y., & Rahman, A. (2022). Machine Learning-Based Automated Diagnostic Systems Developed for Heart Failure Prediction Using Different Types of Data Modalities: A Systematic Review and Future Directions. _Computational and Mathematical Methods in Medicine_ , _2022_ , 1-30. <u>https://doi.org/10.1155/2022/9288452</u> 

672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 

Maaliw, R. R., Alon, A. S., Lagman, A. C., Garcia, M. B., Susa, J. A. B., Reyes, R. C., Fernando-Raguro, M. C., & Hernandez, A. A. (2022). A Multistage Transfer Learning Approach for Acute Lymphoblastic Leukemia Classification. _2022 IEEE 13th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON)_ , 488-495. <u>https://doi.org/10.1109/UEMCON54665.2022.9965679</u> 

- Maaliw, R. R., Susa, J. A. B., Alon, A. S., Lagman, A. C., Ambat, S. C., Garcia, M. B., Piad, K. C., & Raguro, M. C. F.-. (2022). A Deep Learning Approach for Automatic Scoliosis Cobb Angle Identification. _2022 IEEE World AI IoT Congress (AIIoT)_ , 111-117. https://doi.org/10.1109/AIIoT54504.2022.9817290 

Mamun, M., Farjana, A., Mamun, M. A., Ahammed, M. S., & Rahman, M. M. (2022). Heart Failure Survival Prediction Using Machine Learning Algorithm: Am I Safe From Heart Failure? _2022 IEEE World AI IoT Congress (AIIoT)_ , 194-200. https://doi.org/10.1109/AIIoT54504.2022.9817303 

- Manogaran, G., & Lopez, D. (2018). Health Data Analytics Using Scalable Logistic Regression With Stochastic Gradient Descent. _International Journal of Advanced Intelligence Paradigms_ , _10_ (1-2), 118-132. 

<u>https://doi.org/10.1504/IJAIP.2018.089494</u> 

- Methaila, A., Kansal, P., Arya, H., & Kumar, P. (2014). Early Heart Disease Prediction Using Data Mining Techniques. _Fourth International Conference on Computational Science, Engineering and Information Technology_ , 53-59. <u>https://doi.org/10.5121/csit.2014.4807</u> 

Nalavade, J. E., Gavali, M. L., Gohil, N. D., & Jamale, S. C. (2014). Impelling Heart Attack Prediction System using Data Mining and Artificial Neural Network. _International Journal of Current Engineering and Technology_ , _4_ (3), 1575-1579. https://inpressco.com/wp-content/uploads/2014/05/Paper711575-1579.pdf 

- Özbay Karakuş, M., & Er, O. (2022). A Comparative Study on Prediction of Survival Event of Heart Failure Patients Using Machine Learning Algorithms. _Neural Computing and Applications_ , _34_ (16), 13895-13908. <u>https://doi.org/10.1007/s00521-022-07201-9</u> 

- Pal, M., Parija, S., Panda, G., Dhama, K., & Mohapatra, R. K. (2022). Risk Prediction of Cardiovascular Disease Using Machine Learning Classifiers. _Open Medicine_ , _17_ (1), 1100-1113. https://doi.org/10.1515/med-2022-0508 

- Priya, O. S., Srinivas, K., & Yeruva, S. (2022). Heart Failure Prediction Using Classification Methods. _Proceedings of Second International Conference on Advances in Computer Engineering and Communication Systems_ , 545-553. <u>https://doi.org/10.1007/978-981-16-7389-4_53</u> 

Rajendran, R., & Karthi, A. (2022). Heart Disease Prediction Using Entropy Based Feature Engineering and Ensembling of Machine Learning Classifiers. _Expert Systems with Applications_ , _207_ , 1-15. <u>https://doi.org/10.1016/j.eswa.2022.117882</u> 

- Rajkumar, A., & Reena, G. S. (2010). Diagnosis of Heart Disease Using Datamining Algorithm. _Global Journal of Computer Science and Technology_ , _10_ (10), 38-43. 

<u>https://computerresearch.org/index.php/computer/article/view/1028</u> 

- Reddy, R. V. K., Raju, K. P., Kumar, M. J., Sujatha, C. H., & Prakash, P. R. (2016). Prediction of Heart Disease Using Decision Tree Approach. _International Journal of Advanced Research inComputer Science and Software Engineering_ , _6_ (3), 530-532. <u>https://www.researchgate.net/publication/339106269</u> 

- Ruan, Y., Guo, Y., Zheng, Y., Huang, Z., Sun, S., Kowal, P., Shi, Y., & Wu, F. (2018). Cardiovascular Disease (CVD) and Associated Risk Factors Among Older Adults in Six Low-and Middle-income Countries: Results from SAGE Wave 1. _BMC Public Health_ , _18_ (1), 1-13. https://doi.org/10.1186/s12889-018-5653-9 

- Salman, I. (2019). Heart Attack Mortality Prediction: An Application of Machine Learning Methods. _Turkish Journal of Electrical Engineering and Computer Sciences_ , _27_ (6), 4378-4389. https://doi.org/10.3906/elk-1811-4 

- Sandhu, R. K., Jimenez, M. C., Chiuve, S. E., Fitzgerald, K. C., Kenfield, S. A., Tedrow, U. B., & Albert, C. M. (2012). Smoking, Smoking Cessation, and Risk of Sudden Cardiac Death in Women. _Circulation. Arrhythmia and electrophysiology_ , _5_ (6), 1091-1097. https://doi.org/10.1161/CIRCEP.112.975219 

- Singh, Y. K., Sinha, N., & Singh, S. K. (2017). Heart Disease Prediction System Using Random Forest. _Advances in Computing and Data Sciences_ , 613-623. https://doi.org/10.1007/978-981-10-5427-3_63 

- Sultana, M., Haider, A., & Uddin, M. S. (2016). Analysis of Data Mining Techniques for Heart Disease Prediction. _2016 3rd International Conference on Electrical Engineering and Information Communication Technology (ICEEICT)_ , 1-5. <u>https://doi.org/10.1109/CEEICT.2016.7873142</u> 

- Takci, H. (2018). Improvement of Heart Attack Prediction by the Feature Selection Methods. _Turkish Journal of Electrical Engineering and Computer Sciences_ , _26_ (1), 1-10. https://doi.org/10.3906/elk-1611-235 

- World Health Organization. (2021). _Cardiovascular Diseases (CVDs)_ . https://www.who.int/news-room/fact- <u>sheets/detail/cardiovascular-diseases-(cvds)</u> 

725 

# **ADDITIONAL READING** 

El Ouazzani, R., Fattah, M., & Benamar, N. (Eds.). (2022). _AI Applications for Disease Diagnosis and Treatment_ . IGI Global. https://doi.org/10.4018/978-1-6684-2304-2 

726 727 

728 729 730 731 732 733 734 735 736 737 

738 

739 

- Garcia, M. B. (Ed.). (2022). _Socioeconomic Inclusion During an Era of Online Education_ . IGI Global. <u>https://doi.org/10.4018/978-1-6684-4364-4</u> 

- Nijalingappa, P., Kautish, S., Ghonge, M. M., & Ravi, R. V. (Eds.). (2022). _Leveraging AI Technologies for Preventing and Detecting Sudden Cardiac Arrest and Death_ . IGI Global. https://doi.org/10.4018/978-1-7998-8443-9 

- Roy, M., & Gupta, L. R. (Eds.). (2021). _Machine Learning and Data Analytics for Predicting, Managing, and Monitoring Disease_ . IGI Global. https://doi.org/10.4018/978-1-7998-7188-0 

- Suzuki, K. (Ed.). (2012). _Machine Learning in Computer-Aided Diagnosis: Medical Imaging Intelligence and Analysis_ . IGI Global. https://doi.org/10.4018/978-1-4666-0059-1 

- Yadav, D., Bansal, A., Bhatia, M., Hooda, M., & Morato, J. (Eds.). (2021). _Diagnostic Applications of Health Intelligence and Surveillance Systems_ . IGI Global. https://doi.org/10.4018/978-1-7998-6527-8 

# **KEY TERMS AND DEFINITIONS** 

**KEY TERMS AND DEFINITIONS are missing. According to IGI Global:** 

740 _All book chapters must include a list of 7+ key terms and definitions following the references and_ 741 _additional reading lists. Definitions must be written in the chapter author’s own words. Please_ 742 _include these at the end of the paper._ 

- 743 **ABOUT THE AUTHORS** (This will appear on the author index page) 

- 744 

## **Nilamadhab Mishra,** **_VIT Bhopal University_ , India** 

- 745 Nilamadhab Mishra is currently an Assistant Professor in Post at School of Computing, Debre 746 Berhan University, Ethiopia. He has around 15 years of rich global exposure in Academic Teaching 747 & Research. He publishes numerous peer-reviewed research in SCIE & SCOPUS indexed journals & 748 IEEE conference proceedings and serves as a reviewer and editorial member in peer-reviewed 749 Journals and Conferences. Dr. Mishra has received his Doctor of Philosophy (PhD) in Computer 750 Science & Information Engineering from Graduate Institute of Electrical Engineering, College of 751 Engineering, Chang Gung University (a World Ranking University), Taiwan. He involves in 752 academic research by working, as a Journal Editor, as an SCIE & Scopus indexed Journals Referee, 753 as an ISBN Book Author, and as an IEEE Conference Referee. Dr. Mishra has been pro-actively 754 involved with several professional bodies: CSI, ORCID, IAENG, ISROSET, Senior Member of 755 “ASR” (Hong Kong), Senior Member of “IEDRC” (Hong Kong), and Member of “IEEE ". Dr. 

- 756 Mishra’s Research areas incorporate Network Centric Data Management, Data Science: Analytics 757 and Applications, CIoT Big-Data System, and Cognitive Apps Design & Explorations. 

- 758 **Nishq Poorav Desai,** **_VIT Bhopal University_ , India** 

- 759 Nishq Poorav Desai is a UG research scholar at VIT Bhopal University, India. 

- 760 **Abhijay Wadhwani,** **_VIT Bhopal University_ , India** 

- 761 Abhijay Wadhwani is a UG research scholar at VIT Bhopal University, India. 

- 762 **Mohammed Farhan Baluch,** **_VIT Bhopal University_ , India** 

- 763 Mohammed Farhan Baluch is a UG research scholar at VIT Bhopal University, India. 

764 

## **Sample Biography Format:** 

Manuel B. Garcia is a professor of information technology and the founding director of the Educational Innovation and Technology Hub (EdITH) at FEU Institute of Technology, Manila, Philippines. His interdisciplinary research interest includes topics that, individually or collectively, cover the disciplines of education and information technology. 

