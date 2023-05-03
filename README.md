Download Link: https://assignmentchef.com/product/solved-cs-189-introduction-to-machine-learning-hw4
<br>
Deliverables:

<ol>

 <li>Submit your predictions for the test sets to Kaggle as early as possible. Include your Kaggle scores in your write-up (see below). The Kaggle competition for this assignment can be found at:

  <ul>

   <li><a href="https://www.kaggle.com/c/cs189-hw4-wine">https://www.kaggle.com/c/cs189-hw4-wine</a></li>

  </ul></li>

 <li>Submit a PDF of your homework, with an appendix listing all your code, to the Gradescope assignment entitled “Homework 4 Write-Up”. In addition, please include, as your solutions to each coding problem, the specific subset of code relevant to that part of the problem. You may typeset your homework in LaTeX or Word (submit PDF format, not .doc/.docx format) or submit neatly handwritten and scanned solutions. Please start each question on a new page. If there are graphs, include those graphs in the correct sections. Do not put them in an appendix. We need each solution to be self-contained on pages of its own.

  <ul>

   <li>In your write-up, please state with whom you worked on the homework. • In your write-up, please copy the following statement and sign your signature next to it. (Mac Preview and FoxIt PDF Reader, among others, have tools to let you sign a PDF file.) We want to make it <em>extra </em>clear so that no one inadverdently cheats. <em>“I certify that all solutions are entirely in my own words and that I have not looked at another student’s solutions. I have given credit to all external sources I consulted.”</em></li>

  </ul></li>

 <li>Submit all the code needed to reproduce your results to the Gradescope assignment entitled “Homework 4 Code”. Yes, you must submit your code twice: in your PDF write-up following the directions as described above so the readers can easily read it, and once in compilable/interpretable form so the readers can easily run it. Do NOT include any data files we provided. Please include a short file named README listing your name, student ID, and instructions on how to reproduce your results. Please take care that your code doesn’t take up inordinate amounts of time or memory. If your code cannot be executed, your solution cannot be verified.</li>

</ol>

<h1>1          Logistic Regression with Newton’s Method</h1>

Consider sample points <em>X</em><sub>1</sub>, <em>X</em><sub>2</sub>,…, <em>X<sub>n </sub></em>∈ R<em><sup>d </sup></em>and associated values <em>y</em><sub>1</sub>,<em>y</em><sub>2</sub>,…,<em>y<sub>n </sub></em>∈ {0,1}, an <em>n </em>× <em>d </em>design matrix <em>X </em>= [<em>X</em><sub>1 </sub>… <em>X<sub>n</sub></em>]<sup>&gt; </sup>and an <em>n</em>-vector <em>y </em>= [<em>y</em><sub>1 </sub>… <em>y<sub>n</sub></em>]<sup>&gt;</sup>.

If we add `<sub>2</sub>-regularization to logistic regression, the cost function is

<em><sup>n </sup></em>                                          

X

<em>J</em>(<em>w</em>) = λ |<em>w</em>|2 − <em>y</em><em>i </em>ln <em>s</em><em>i </em>+ (1 − <em>y</em><em>i</em>)ln(1 − <em>s</em><em>i</em>)

<em>i</em>=1

where <em>s<sub>i </sub></em>= <em>s</em>(<em>X<sub>i </sub></em>· <em>w</em>), <em>s</em>(γ) = 1/(1 + <em>e</em><sup>−γ</sup>), and λ &gt; 0 is the regularization parameter. As in lecture, the vector <em>s </em>= [<em>s</em><sub>1 </sub>…         <em>s<sub>n</sub></em>]<sup>&gt; </sup>is a useful shorthand.

In this problem, you will use Newton’s method to minimize this cost function on the four-point, two-dimensional training set

                                         

<em>X </em>= 1010 1331,            <em>y </em>= 1001.

You may want to draw these points on paper to see what they look like. The <em>y</em>-vector implies that the first two sample points are in class 1, and the last two are in class 0.

These sample points cannot be separated by a linear decision boundary that passes <em>through the origin</em>. As described in lecture, append a 1 to each <em>X<sub>i </sub></em>vector and use a weight vector <em>w </em>∈ R<sup>3 </sup>whose last component is the bias term (the term we call α in lecture).

<ol>

 <li>Derive the gradient of the cost function <em>J</em>(<em>w</em>). Your final answer should be a simple matrixvector expression. While you may derive the matrix-vector form by first deriving the components of the gradient vector, do NOT write your answer in terms of these individual components.</li>

 <li>Derive the Hessian of <em>J</em>(<em>w</em>). Again, your answer should be a simple matrix-vector expression.</li>

 <li>State the update equation for one iteration of Newton’s method for this problem.</li>

</ol>

&gt;

<ol start="4">

 <li>We are given a regularization parameter of λ = 0.07 and a starting point of <em>w</em><sup>(0) </sup>= <sup>h</sup>−2 1 0<sup>i </sup>. For the following four parts, you need only state the final solution. Thus you may derive the solution by hand or implement Newton’s algorithm and report the final result. If you do the latter, you do not need to submit code for this part.

  <ul>

   <li>State the value of <em>s</em><sup>(0) </sup>(the value of <em>s </em>before any iterations).</li>

   <li>State the value of <em>w</em><sup>(1) </sup>(the value of <em>w </em>after one iteration).</li>

   <li>State the value of <em>s</em><sup>(1)</sup>.</li>

   <li>State the value of <em>w</em><sup>(2) </sup>(the value of <em>w </em>after two iterations).</li>

  </ul></li>

</ol>

<h1>2          `<sub>1</sub>– and `<sub>2</sub>-Regularization</h1>

Consider sample points <em>X</em><sub>1</sub>, <em>X</em><sub>2</sub>,…, <em>X<sub>n </sub></em>∈ R<em><sup>d </sup></em>and associated values <em>y</em><sub>1</sub>,<em>y</em><sub>2</sub>,…,<em>y<sub>n </sub></em>∈ R, an <em>n</em>×<em>d </em>design matrix <em>X </em>= [<em>X</em><sub>1 </sub>… <em>X<sub>n</sub></em>]<sup>&gt; </sup>and an <em>n</em>-vector <em>y </em>= [<em>y</em><sub>1 </sub>… <em>y<sub>n</sub></em>]<sup>&gt;</sup>. For the sake of simplicity, assume that the sample data has been centered and whitened so that each feature has mean 0 and variance

&gt;

1 and the features are uncorrelated; i.e., <em>X X </em>= <em>nI</em>. For this question, we will not use a fictitious dimension nor a bias term; our linear regression function will output zero for <em>x </em>= 0.

Consider linear least-squares regression with regularization in the `<sub>1</sub>-norm, also known as Lasso. The Lasso cost function is

<em>J</em>(<em>w</em>) = |<em>Xw </em>− <em>y</em>|<sup>2 </sup>+ λ k<em>w</em>k<sub>1</sub>

where <em>w </em>∈ R<em><sup>d </sup></em>and λ &gt; 0 is the regularization parameter. Let <em>w</em><sup>∗ </sup>= argmin <em><sub>w</sub></em><sub>∈R</sub><em>d J</em>(<em>w</em>) denote the weights that minimize the cost function.

In the following steps, we will show that whitened training data decouples the features, so that <em>w</em><sup>∗</sup><em><sub>i </sub></em>is determined by the <em>i</em><sup>th </sup>feature alone (i.e., column <em>i </em>of the design matrix <em>X</em>), regardless of the other features. This is true for both Lasso and ridge regression.

<ol>

 <li>We use the notation <em>X</em><sub>∗<em>i </em></sub>to denote column <em>i </em>of the design matrix <em>X</em>, which represents the <em>i</em><sup>th </sup> Write <em>J</em>(<em>w</em>) in the following form for appropriate functions <em>g </em>and <em>f</em>.</li>

</ol>

<em>d</em>

X

<em>J</em>(<em>w</em>) = <em>g</em>(<em>y</em>) +           <em>f</em>(<em>X</em><sub>∗<em>i</em></sub>,<em>w<sub>i</sub></em>,<em>y</em>,λ)

<em>i</em>=1

<ol start="2">

 <li>If <em>w</em><sup>∗</sup><em><sub>i </sub></em>&gt; 0, what is the value of <em>w</em><sup>∗</sup><em><sub>i </sub></em>?</li>

 <li>If <em>w</em><sup>∗</sup><em><sub>i </sub></em>&lt; 0, what is the value of <em>w</em><sup>∗</sup><em><sub>i </sub></em>?</li>

 <li>Considering parts 2 and 3, what is the condition for <em>w</em><sup>∗</sup><em><sub>i </sub></em>to be zero?</li>

 <li>Now consider ridge regression, which uses the `<sub>2 </sub>regularization term λ |<em>w</em>|<sup>2</sup>. How does this change the function <em>f</em>(·) from part 1? What is the new condition in which <em>w</em><sup>∗</sup><em><sub>i </sub></em>= 0? How does it differ from the condition you obtained in part 4?</li>

</ol>

<h1>3          Regression and Dual Solutions</h1>

<ol>

 <li>For a vector <em>w</em>, derive ∇ |<em>w</em>|<sup>4</sup>. Then derive ∇<em><sub>w </sub></em>|<em>Xw </em>− <em>y</em>|<sup>4</sup>.</li>

 <li>Consider sample points <em>X</em><sub>1</sub>, <em>X</em><sub>2</sub>,…, <em>X<sub>n </sub></em>∈ R<em><sup>d </sup></em>and associated values <em>y</em><sub>1</sub>,<em>y</em><sub>2</sub>,…,<em>y<sub>n </sub></em>∈ R, an <em>n </em>× <em>d </em>design matrix <em>X </em>= [<em>X</em><sub>1 </sub>… <em>X<sub>n</sub></em>]<sup>&gt; </sup>and an <em>n</em>-vector <em>y </em>= [<em>y</em><sub>1 </sub>… <em>y<sub>n</sub></em>]<sup>&gt;</sup>, and the regularized regression problem</li>

</ol>

<em>w</em><sup>∗ </sup>= argmin |<em>Xw </em>− <em>y</em>|<sup>4 </sup>+ λ |<em>w</em>|<sup>2</sup>,

<em>w</em>∈R<em>d</em>

which is similar to ridge regression, but we take the fourth power of the error instead of the squared error. (It is not possible to write the optimal solution <em>w</em><sup>∗ </sup>as the solution of a system of linear equations, but it can be found by gradient descent or Newton’s method.)

Show that the optimum <em>w</em><sup>∗ </sup>is unique. By setting the gradient of the objective function to zero, show that <em>w</em><sup>∗ </sup>can be written as a linear combination <em>w</em><sup>∗ </sup>= <sup>P<em>n</em></sup><em><sub>i</sub></em><sub>=1 </sub><em>a<sub>i</sub>X<sub>i </sub></em>for some scalars <em>a</em><sub>1</sub>,…,<em>a<sub>n</sub></em>. Write the vector <em>a </em>of dual coefficients in terms of <em>X</em>, <em>y</em>, λ, and the optimal solution <em>w</em>∗.

<ol start="3">

 <li>Consider the regularized regression problem</li>

</ol>

<em>n</em>

1

∗ X &gt; ,<em>y<sub>i</sub></em>) + λ |<em>w</em>|2 <em>w </em>= argmin <em>L</em>(<em>w X<sub>i</sub></em>

<em>w</em>∈R<em>d             n </em><em>i</em>=1

where the cost function <em>L </em>is convex in its first argument. Prove that the optimal solution has the form <em>w</em><sup>∗ </sup>= <sup>P<em>n</em></sup><em><sub>i</sub></em><sub>=1 </sub><em>a<sub>i</sub>X<sub>i</sub></em>. If the cost function is not convex, does the optimal solution always have the form <em>w</em><sup>∗ </sup>= <sup>P<em>n</em></sup><em><sub>i</sub></em><sub>=1 </sub><em>a<sub>i</sub>X<sub>i</sub></em>? Justify your answer.

<h1>4          Wine Classification with Logistic Regression</h1>

The wine dataset given to you as part of the homework in data.mat consists of 6,497 data points, each having 12 features. The description of these features is provided in data.mat. The dataset includes a training set of 6,000 data points and a test set of 497 data points. Your classifier needs to predict whether a wine is white (class label 0) or red (class label 1).

For this homework, you need to do the following.

<ol>

 <li>Derive and write down the batch gradient descent update equation for logistic regression with `<sub>2 </sub> (Not Newton’s method, but the slower batch gradient descent.)</li>

</ol>

Choose a reasonable regularization parameter value and a reasonable learning rate. Run your algorithm and plot the cost function as a function of the number of iterations. (As this is batch descent, one “iteration” should use every sample point once.)

<ol start="2">

 <li>Derive and write down the stochastic gradient descent update equation for logistic regression with `<sub>2 </sub> Choose a suitable learning rate. Run your algorithm and plot the cost function as a function of the number of iterations—where now each “iteration” uses <em>just one </em>sample point.</li>

</ol>

Comment on the differences between the convergence of batch and stochastic gradient descent.

<ol start="3">

 <li>Instead of a constant learning rate , repeat part 2 where the learning rate decreases as ∝ 1/<em>t </em>for the <em>t</em><sup>th </sup> Plot the cost function vs. the number of iterations. Is this strategy better than having a constant ?</li>

 <li>Finally, train your classifier on the entire training set. Submit your results to Kaggle. Your classifier, when given the test points, should output a CSV file. (There is a sample one on Kaggle.) You’ll upload this CSV file to Kaggle where it’ll be scored with both a public test set, and a private test set. You will be able to see only your public score. You can only submit twice per day, so get started early! In your writeup, for this problem, report your Kaggle username, the best score you achieved on Kaggle, and a short writeup describing what you did to achieve that score.</li>

</ol>

IMPORTANT: Do NOT use any software package for logistic regression that you didn’t write yourself!

<h1>5          Real World Spam Classification</h1>

Motivation: After taking CS 189 or CS 289A, students should be able to wrestle with “real-world” data and problems. These issues might be deeply technical and require a theoretical background, or might demand specific domain knowledge. Here is an example that a past TA encountered.

Daniel (a past CS 189 TA) interned as an anti-spam product manager for an email service provider. His company uses a linear SVM to predict whether an incoming spam message is spam or ham. He notices that the number of spam messages received tends to spike upwards a few minutes before and after midnight. Eager to obtain a return offer, he adds the timestamp of the received message, stored as number of milliseconds since the previous midnight, to each feature vector for the SVM to train on, in hopes that the ML model will identify the abnormal spike in spam volume at night. To his dismay, after testing with the new feature, Daniel discovers that the linear SVM’s success rate barely improves.

Why can’t the linear SVM utilize the new feature well, and what can Daniel do to improve his results? Daniel is unfortunately limited to only a quadratic kernel. This is an actual interview question Daniel received for a machine learning engineering position!

Write a short explanation. This question is open ended and there can be many correct answers.