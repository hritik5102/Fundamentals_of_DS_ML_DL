## **Neural network concept**

---
### **`Shallo neural network`**
---    
"shallow" neural networks is a term used to describe neural network that usually have a only one hidden layer as opposed to deep neural network which has several hidden layers, often of various types.

[<img src="https://qphs.fs.quoracdn.net/main-qimg-257af1d7bfdc2d7c1c4f4c30366a3c77.webp" width="70%" />](https://qphs.fs.quoracdn.net/main-qimg-257af1d7bfdc2d7c1c4f4c30366a3c77.webp)

---
### **`Perceptron`**
---
* Perceptron is the basic unit of a neural network.<br/>

* The perceptron is a network takes a number of inputs,
carries out some processing on these inputs and produces as output.

* Artificial Neural Networks (ANN) are comprised of a large number of simple elements , called neurons, each of which makes simple decisions. Together, the neurons can provide accurate answers to some complex problems, such as natural language processing, computer vision, and AI.

  **Figure** :
  
  <img src="https://qph.fs.quoracdn.net/main-qimg-1a057e476f5c069f825fa198780c211b.webp" width="50%"/>

* **`What are significance of hidden layer ?`**

  **Answer** : without a hidden layer . it's like single neuron i.e. perceptron or like logistic or svm 
  single neuron can only implement a linear boundary.
  if we want a our network to classify a non linear boundary then we have to use hidden layer. 

* **`what are the hyper parameters ?`**

  **Answer** : number of the hidden layer and number of neurons in each hidden layer 
  will be the hyper parameters

* Increasing number of hidden layer leads to overfitting.


---
### **`Universal approximation theorem`**
---

**_Defination_** : 
A feedforward network with a single layer is sufficient to represent any function, but the layer may be infeasibly large and may fail to learn and generalize correctly.

That simply means that we can solve any non trivial function or continous function using only single hidden layer which contain n number of neurons in it with help of activation function.<br/>

**`Statement`**:
> Introducing non-linearity via an activation function allows us to approximate any function. It’s quite simple, really. — Elon Musk 

It has been proven that  non-trivial function (x³ + x² - x -1) using a single hidden layer and 6 neurons <br/>

#### **Type of activation function** :

1. Linear or Identity Activation Function
   
2. Non-linear Activation Function
     * Logistic Function or sigmoid function
     * Tanh (hyperbolic tangent Activation) function
     * ReLu ( Rectified Linear Units )
     * Leaky ReLu
     * Softmax function

We can use any activation function , now let us choose activation function as ReLu to represent the above non trivial function.

<img src="https://miro.medium.com/max/1400/1*6c0BjULsvVe5zqlkB_Vb3w.png
" width="50%"/>

We chose x³+x²-x -1 as my target function. Using only ReLU max(0,x), we iteratively tried different combinations of ReLUs until we had an output that roughly resembled the target. 

Here are the results I achieved taking the weighted sum of 3 ReLUs.<br/>

[<img src="https://miro.medium.com/max/1400/1*qt4SaoYphChAreRTDJewIw.png" width="50%"/>](https://miro.medium.com/max/1400/1*qt4SaoYphChAreRTDJewIw.png)

So combining 3 ReLU functions is like training a network of 3 hidden neurons. Here are the equations I used to generate these charts.

[<img src="https://miro.medium.com/max/1000/1*fdICiWJocvOTJPoRrhek_Q.png" width="50%"/>](https://miro.medium.com/max/1000/1*fdICiWJocvOTJPoRrhek_Q.png)

Each neuron’s output equals ReLU wrapped around the weighted input wx + b.

I found I could shift the ReLU function left and right by changing the bias and adjust the slope by changing the weight. I combined these 3 functions into a final sum of weighted inputs (Z) which is standard practice in most neural networks.

The negative signs in Z represent the final layer’s weights which I set to -1 in order to “flip” the graph across the x-axis to match our concave target. After playing around a bit more I finally arrived at the following 7 equations that, together, roughly approximate x³+x²-x -1.

[<img src="https://miro.medium.com/max/1400/1*lihbPNQgl7oKjpCsmzPDKw.png" width="50%"/>](https://miro.medium.com/max/1400/1*lihbPNQgl7oKjpCsmzPDKw.png).

Hard-coding my weights into a real network
Here is a diagram of a neural network initialized with my fake weights and biases. If you give this network a dataset that resembles x³+x²-x-1, it should be able approximate the correct output for inputs between -2 and 2.

[<img src="https://miro.medium.com/max/1400/1*RWcNXtQSrIVoiw99bkcA8w.png" width="50%"/>](https://miro.medium.com/max/1400/1*RWcNXtQSrIVoiw99bkcA8w.png)

That last statement, approximate the correct output for any input between -2 and 2, is key. The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range.

Refer this : [https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6)

---
## **`Forward propagation in neural networks`**
---
A learning algorithm/model finds out the parameters (weights and biases) with the help of forward propagation and backpropagation.

The input data is fed in the forward direction through the network. Each hidden layer accepts the input data, processes it as per the activation function and passes to the successive layer.

The feed-forward network helps in forward propagation.<br/>

At each neuron in a hidden or output layer, the processing happens in two steps:

* **`Preactivation`** : it is a weighted sum of inputs i.e. the linear transformation of weights w.r.t to inputs available. Based on this aggregated sum and activation function the neuron makes a decision whether to pass this information further or not.<br/>
  **Eg** :  We compute F(x) = (x * wx + y * wy+b) at each neuron in hidden layer

* **`Activation`** : the calculated weighted sum of inputs is passed to the activation function. An activation function is a mathematical function which adds non-linearity to the network. There are four commonly used and popular activation functions — sigmoid, hyperbolic tangent(tanh), ReLU and Softmax.

    [<img src="https://www.dspguide.com/graphics/F_26_6.gif" width="60%"/>](https://www.dspguide.com/graphics/F_26_6.gif)

  * we apply Activation function on these function (f(x)) , which will give a some output.

    if Activation function output is not equal to Target output then we apply weight updated.

* **`Weight update`** :
  
    ---

      alpha : learning rate (Larget alpha that means we are making larger changes to the weight.) 
      t : target output 
      i : Data points 
      p(i) : output of activation function  
   
   w(new) = w(old) + alpha * (t - p(i)) * i  <br/>

  ---

  Weights on the edge is decided based on the :<br/>
      
      number of input neuron i.e. (d) and
      number of neurons on hidden layer i.e.( n )   
      hence we have (d * n) number of edge with corresponding weight. 

---

The data can be generated using make_moons() function of sklearn.datasets module. The total number of samples to be generated and noise about the moon’s shape can be adjusted using the function parameters.

```python
      import numpy as np
      import matplotlib.pyplot as plt
      import matplotlib.colors
      from sklearn.datasets import make_moonsnp.random.seed(0)data, labels = make_moons(n_samples=200,noise = 0.04,random_state=0)
      print(data.shape, labels.shape)color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow"])
      plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap)
      plt.show()
```

<br/>

[<img src="https://miro.medium.com/max/924/1*F50x2COgQ8LySWV0LtdFLw.png" width="40%"/>](https://miro.medium.com/max/924/1*F50x2COgQ8LySWV0LtdFLw.png)

Here, 200 samples are used to generate the data and it has two classes shown in red and green color.

Now, let us see the neural network structure to predict the class for this binary classification problem. Here, I am going to use one hidden layer with two neurons, an output layer with a single neuron and sigmoid activation function.

[<img src="https://miro.medium.com/max/1192/1*tp73P0isrrfpj8RG-5aH6w.png" width="40%"/>](https://miro.medium.com/max/1192/1*tp73P0isrrfpj8RG-5aH6w.png)

During forward propagation at each node of hidden and output layer preactivation and activation takes place. For example at the first node of the hidden layer, a1(preactivation) is calculated first and then h1(activation) is calculated.

a1 is a weighted sum of inputs. Here, the weights are randomly generated.

a1 = w1 * x1 + w2 * x2 + b1 = 1.76 * 0.88 + 0.40 *(-0.49) + 0 = 1.37 approx and <br/> h1 is the value of activation function applied on a1.

[<img src="https://miro.medium.com/max/576/1*WrkgXLQSjHpzmR_H3xsnCQ.png" width="30%"/>](https://miro.medium.com/max/576/1*WrkgXLQSjHpzmR_H3xsnCQ.png
)

Similarly

a2 = w3*x1 + w4*x2 + b2 = 0.97 *0.88 + 2.24 *(- 0.49)+ 0 = -2.29 approx and

[<img src="https://miro.medium.com/max/618/1*46xma79g8Gdew_LbT6x2aw.png" width="30%"/>](https://miro.medium.com/max/618/1*46xma79g8Gdew_LbT6x2aw.png)

For any layer after the first hidden layer, the input is output from the previous layer.

a3 = w5*h1 + w6*h2 + b3 = 1.86*0.8 + (-0.97)*0.44 + 0 = 1.1 approx
and

[<img src="https://miro.medium.com/max/598/1*lCVQROFldjILndKg-pKHxw.png" width="30%"/>](https://miro.medium.com/max/598/1*lCVQROFldjILndKg-pKHxw.png)

So there are 74% chances the first observation will belong to class 1. Like this for all the other observations predicted output can be calculated.

[<img src="https://miro.medium.com/max/1400/1*ts5LSdtkfSsMYS7M0X84Tw.gif" width="60%"/>](https://miro.medium.com/max/1400/1*ts5LSdtkfSsMYS7M0X84Tw.gif)

**Reference** : 
[https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250](https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250)

---
## **`Activation function`**
---

* While building a neural network, one of the mandatory choices we need to make is 
which activation function to use. In fact, it is an unavoidable choice because 
activation functions are the foundations for a neural network to learn and 
approximate any kind of complex and continuous relationship between variables. 
It simply adds non-linearity to the network.

  <img src="https://image.ibb.co/gEmoSQ/mmm_act_function_1.png
  " width="50%"/>

* Activation functions reside within neurons, but not all neurons.
Hidden and output layer neurons possess activation functions, but input layer neurons do not.

* Activation functions perform a transformation on the input received, 
in order to keep values within a manageable range. Since values in the 
input layers are generally centered around zero and have already been appropriately scaled,
they do not require transformation. However, these values, once multiplied by weights and 
summed, quickly get beyond the range of their original scale, which is where the activation 
functions come into play, forcing values back within this acceptable range and making them 
useful.

* In order to be useful, activation functions must also be nonlinear and continuously differentiable.
Nonlinearity allows the neural network to be a universal approximation; As we already discuss. A continuously differentiable function is necessary for gradient-based optimization methods, 
which is what allows the efficient back propagation of errors throughout the network.

  **`NOTE`**:
  Inside the neuron:

    * An activation function is assigned to the neuron or entire layer of neurons.
    * weighted sum of input values are added up.
    * the activation function is applied to weighted sum of input values and transformation takes place. 
    * Activation functions also help normalize the output of each neuron to a 
   range between 1 and 0 or between -1 and 1.
    * the output to the next layer consists of this transformed value. 
  <br/>

**Reference** :

* [https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)

* [https://towardsdatascience.com/analyzing-different-types-of-activation-functions-in-neural-networks-which-one-to-prefer-e11649256209](https://towardsdatascience.com/analyzing-different-types-of-activation-functions-in-neural-networks-which-one-to-prefer-e11649256209)

* [https://cs231n.github.io/neural-networks-1/#actfun](https://cs231n.github.io/neural-networks-1/#actfun)

* [https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer](https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer)

---
## **`Types of Activation Functions`**
---

The Activation Functions can be basically divided into 3 types-

 * Binary Step Function
 * Linear Activation Function
 * Non-linear Activation Functions

<hr/>

#### 1. **Binary Step Function**

  A binary step function is a threshold-based activation function. If the input value is above or below a certain threshold, the neuron is activated and sends exactly the same signal to the next layer.

  [<img src="https://missinglink.ai/wp-content/uploads/2018/11/binarystepfunction.png" width="40%"/>](https://missinglink.ai/wp-content/uploads/2018/11/binarystepfunction.png)

  The problem with a step function is that it does not allow multi-value outputs—for example, it cannot support classifying the inputs into one of several categories.

#### 2. **Linear or Identity Activation Function**

  It takes the inputs, multiplied by the weights for each neuron, 
  and creates an output signal proportional to the input. In one sense, 
  a linear function is better than a step function because it allows multiple outputs,
  not just yes and no.

  A linear activation function takes the form: A = cx

  [<img src="https://missinglink.ai/wp-content/uploads/2018/11/graphsright.png" width="30%"/>](https://missinglink.ai/wp-content/uploads/2018/11/graphsright.png)

  However, a linear activation function has two major problems:

  * `Not possible to use backpropagation (gradient descent)` to train the model—the derivative of the function is a constant, and has no relation to the input, X. So it’s not possible to go back and understand which weights in the input neurons can provide a better prediction.

  * `All layers of the neural network collapse into one` —with linear activation functions, no matter how many layers in the neural network, the last layer will be a linear function of the first layer (because a linear combination of linear functions is still a linear function). So a linear activation function turns the neural network into just one layer.

  #### **`Example`**

  <img src="https://miro.medium.com/max/1338/1*xcBdSYRndl6dhouE1y0KHg.png" width="50%"/>
    
  <br/>

  Consider a case where no activation function is used in this network, then from the hidden layer 1 the calculated weighted sum of inputs will be directly passed to hidden layer 2 and it calculates a weighted sum of inputs and pass to the output layer and it calculates a weighted sum of inputs to produce the output. The output can be presented as 
  <br/>

  <img src="https://miro.medium.com/max/1068/1*9es-pAjxSJe3tN61B6p64A.png
  " width="50%"/>

  So the output is simply a linear transformation of weights and inputs and it is not adding any non-linearity to the network. Therefore, this network is similar to a linear regression model which can only address the linear relationship between variables i.e. a model with limited power and not suitable for complex problems like image classifications, object detections, language translations, etc.


#### 3. **Non-Linear Activation Functions**

  Modern neural network models use non-linear activation functions. 
  They allow the model to create complex mappings between the network’s inputs and outputs,
  which are essential for learning and modeling complex data, such as images, video, audio, 
  and data sets which are non-linear or have high dimensionality. 

  Non-linear functions address the problems of a linear activation function:

  1. They allow backpropagation because they have a derivative function which is related to the inputs.
  2. They allow “stacking” of multiple layers of neurons to create a deep neural network.

---
## **`Type of Nonlinear Activation Functions  and How to Choose an Activation Function`**
---

    1. Sigmoid / Logistic
    2. tanh ( hyperbolic tangent)
    3. ReLu (Rectified linear units)
    4. Leaky ReLu
    5. Softmax activation function

* **Sigmoid / Logistic Activation function**
  
  It is a “S” shaped curve with equation : <br/>
  
  [<img src="https://miro.medium.com/max/152/1*2MoOSKaUQyj0_9Q-lnVkEA.png"/>](https://miro.medium.com/max/152/1*2MoOSKaUQyj0_9Q-lnVkEA.png)
  
  [<img src="https://miro.medium.com/max/920/1*qRS650xg0-JrXJPUD_E32w.png" width="30%"/>](https://miro.medium.com/max/920/1*qRS650xg0-JrXJPUD_E32w.png)

  **Advantages**

    Smooth gradient, preventing “jumps” in output values.
    Output values bound between 0 and 1, normalizing the output of each neuron.
    Clear predictions — For X above 2 or below -2, tends to bring the Y value (the prediction) to the edge of the curve, very close to 1 or 0. This enables clear predictions.

  **Disadvantages**

    Vanishing gradient—for very high or very low values of X, there is almost no change to the prediction, causing a vanishing gradient problem. This can result in the network refusing to learn further, or being too slow to reach an accurate prediction.
      
   1. Outputs not zero centered.
   2. Computationally expensive

* **Tanh (Hyperbolic tangent) Function**  
   
    It is similar to logistic activation function with a mathematical equation.<br/>

    [<img src="https://miro.medium.com/max/1230/1*ibDdAN-lHnSafuCG1EjP6g.png"/>](https://miro.medium.com/max/1230/1*ibDdAN-lHnSafuCG1EjP6g.png)

    [<img src="https://miro.medium.com/max/964/1*IrLb4Z_Mp-cbyCa6bBgsKg.png" width="50%"/>](https://miro.medium.com/max/964/1*IrLb4Z_Mp-cbyCa6bBgsKg.png)

  The output ranges from -1 to 1 and having an equal mass on both the sides of zero-axis so it is zero centered function. So tanh overcomes the non-zero centric issue of the logistic activation function. Hence optimization becomes comparatively easier than logistic and it is always preferred over logistic.

  But Still, a tanh activated neuron may lead to saturation and cause vanishing gradient problem.

  The derivative of tanh activation function.

  [<img src="https://miro.medium.com/max/1216/1*ZyQv9ma0lFipjC3vRwSRcw.png" width="50%"/>](https://miro.medium.com/max/1216/1*ZyQv9ma0lFipjC3vRwSRcw.png)

  Issues with tanh activation function:

      Saturated tanh neuron causes the gradient to vanish.
      Because of e^x, it is highly compute-intensive.

* ReLu (Rectified linear units)

  It is the most commonly used function because of its simplicity. It is defined as <br/>

    [<img src="https://miro.medium.com/max/764/1*jyyzxadG8Sbqcgfv08ttAw.png" width="50%"/>](https://miro.medium.com/max/764/1*jyyzxadG8Sbqcgfv08ttAw.png)

  If the input is a positive number the function returns the number itself and if the input is a negative number then the function returns 0.

  [<img src="https://miro.medium.com/max/802/1*E9Az5dBreEwvI5JmlG1dTA.png" width="50%"/>](https://miro.medium.com/max/802/1*E9Az5dBreEwvI5JmlG1dTA.png)

  The derivative of ReLu activation function is given as : <br/>

  [<img src="https://miro.medium.com/max/636/1*vDV1QZKWD3MoS96Ht2GknA.png" width="50%"/>](https://miro.medium.com/max/636/1*vDV1QZKWD3MoS96Ht2GknA.png
)

  **Advantages of ReLu activation function**:<br/>
    
    * Easy to compute.
    * Does not saturate for the positive value of the weighted sum of inputs.

    Because of its simplicity, ReLu is used as a standard activation function in CNN.

    But still, ReLu is not a zero-centered function.

    **`Issues with ReLu activation function`**

    ReLu is defined as max(0, w1x1 + w2x2 + …+b) <br/>

    Now Consider a case b(bias) takes on (or initialized to) a large negative value then the weighted sum of inputs is close to 0 and the neuron is not activated. <br/> That means the ReLu activation neuron dies now. Like this, up to 50% of ReLu activated neurons may die during the training phase.

    To overcome this problem, two solutions can be proposed

      1. Initialize the bias(b) to a large positive value.
      2. Use another variant of ReLu known as Leaky ReLu.
  
* **Leaky ReLu**

    It was proposed to fix the dying neurons problem of ReLu. It introduces a small slope to keep the update alive for the neurons where the weighted sum of inputs is negative. It is defined as

    [<img src="https://miro.medium.com/max/924/1*sy8LauNPCdU6ycPFuqhKag.png" width="50%"/>](https://miro.medium.com/max/924/1*sy8LauNPCdU6ycPFuqhKag.png)

    If the input is a positive number the function returns the number itself and if the input is a negative number then it returns a negative value scaled by 0.01(or any other small value).

    [<img src="https://miro.medium.com/max/860/1*2U9o7ma_pp4rkalyMs2oXg.png" width="50%"/>](https://miro.medium.com/max/860/1*2U9o7ma_pp4rkalyMs2oXg.png)

  The derivative of LeakyReLu is given as:

  [<img src="https://miro.medium.com/max/712/1*nyX9El4tLjP2XdtaYsyxJQ.png" width="50%"/>](https://miro.medium.com/max/712/1*nyX9El4tLjP2XdtaYsyxJQ.png)
  
    **Advantages of LeakyReLu** <br/>
        1. No saturation problem in both positive and negative region.<br/>
        2. The neurons do not die because 0.01x ensures that at least a small gradient will flow through. Although the change in weight will be small but after a few iterations it may come out from its original value.<br/>
        3. Easy to compute.<br/>
        4. Close to zero-centered functions


* **Softmax activation function**

  * Where other activation function get a input value and transform it.
    where softmax consider the information about the whole set of information that means
    it's a special function , where each element in output depends on the entire set
    of element of the input
    
  * Suppose we have input X , weight and bias 
    now in hidden layer , each neuron compute
      1. a = x*w + b <br/>
      2. activation function(a)

  * Now we can apply any activation function let us apply a softmax function

  * Suppose we get value of a = [-0.21 , 0.47 , 1.72 ] 

    <a href="https://ibb.co/dmnyS3R"><img src="https://i.ibb.co/2ZXBwLT/Screenshot-329.png" alt="Screenshot-329" width="70%" border="0"></a>

  * Now we apply softmax function on it. 
  * Note : we could even normalize by dividing perticular number by sum of all number , so why are using exponent , so simple answer is exponent ensure positivity i.e. even negative number becomes positive.
  * Property of softmax :
        1. Normalize the value
        2. Ranges between 0 to 1
        3. sum of value = 1
        4. Probability of each neuron 

    <a href="https://ibb.co/kJNwHpC"><img src="https://i.ibb.co/0KH6Dkb/Screenshot-327.png" alt="Screenshot-327" width="70%" border="0"></a>

  * As we see we get value 0.1,0.2 and 0.7 after apply softmax 
  * so now we can say that 0.7 would be highest probable value. As input was "horse" image
  so output probability of horse is 0.7. 

    <a href="https://ibb.co/7VVn9Lj"><img src="https://i.ibb.co/wCCSvjB/Screenshot-328.png" alt="Screenshot-328" width="70%" border="0"></a>

    **Figure describing softmax funtion** :

    [<img src="https://developers.google.com/machine-learning/crash-course/images/SoftmaxLayer.svg" width="50%"/>](https://developers.google.com/machine-learning/crash-course/images/SoftmaxLayer.svg)

**Note** :

Sigmoid can be used in output layer , but it only ensure that value will be between 0 to 1 , but sum of these value would not be equal to 1 . which does not satisfying the principle of probability distribution.

Softmax acivation function is mostly used in output layer only and not in hidden layer .

Refer to this stack overflow for better explanation : <br/>

[Why use softmax only in the output layer and not in hidden layers?](https://stackoverflow.com/questions/37588632/why-use-softmax-only-in-the-output-layer-and-not-in-hidden-layers)

---

**Why derivative/differentiation is used ?**

When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.That is why we use differentiation in almost every part of Machine Learning and Deep Learning.

[<img src="https://miro.medium.com/max/1400/1*p_hyqAtyI8pbt2kEl6siOQ.png" width="50%"/>](https://miro.medium.com/max/1400/1*p_hyqAtyI8pbt2kEl6siOQ.png)

[<img src="https://miro.medium.com/max/1400/1*n1HFBpwv21FCAzGjmWt1sg.png" width="50%" />](https://miro.medium.com/max/1400/1*n1HFBpwv21FCAzGjmWt1sg.png)

## **`End Notes: Now which one to prefer?`**

* As a rule of thumb, you can start with ReLu as a general approximator and switch to other functions if ReLu doesn't provide better results.
    
* For CNN, ReLu is treated as a standard activation function but if it suffers from dead neurons then switch to LeakyReLu.
    
* Always remember ReLu should be only used in hidden layers.
For classification, Sigmoid functions(Logistic, tanh, Softmax) and their combinations work well. But at the same time, it may suffer from vanishing gradient problem.

* For RNN, the tanh activation function is preferred as a standard activation function.

* For regression i.e. for real value <br/>
      output activation - Linear activation
      Loss function - MSE

  For classification i.e for probability <br/>
      output activation - softmax activation
      Loss function - Cross entropy


---

**Reference** : 

1. [https://www.youtube.com/watch?v=p-XCC0y8eeY](https://www.youtube.com/watch?v=p-XCC0y8eeY)

2. [https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

3. [https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer?newreg=c9a3c1990ce2419785c5c368ecb77d2e](https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer?newreg=c9a3c1990ce2419785c5c368ecb77d2e
)

---
## **`So how neural network learn.`**
---

* Suppose , consider the output layer where we get the probabillity of each classes to be predicted using softmax activation function.

* Now we have label for perticular input.
* we compute a cost based on : (predicted - desired output)^2
* We keep minimizing the cost 

    <a href="https://ibb.co/pzjv1rb"><img src="https://i.ibb.co/zRVhmHQ/Screenshot-331.png" alt="Screenshot-331" width="50%" border="0"></a>

References: 

* [https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)

---
## **`Epoch vs Batch Size vs Iterations vs Learning rate`**
---

* ### **Epoch** 

      One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
                                  OR      
      Number of epochs is the number of times the whole training data is shown to the network while training.

    Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.

    **`Why we use more than one Epoch?`**

    I know it doesn’t make sense in the starting that — passing the entire dataset through a neural network is not enough. And we need to pass the full dataset multiple times to the same neural network. But keep in mind that we are using a limited dataset and to optimise the learning and the graph we are using Gradient Descent which is an iterative process. So, updating the weights with single pass or one epoch is not enough.

    > One epoch leads to underfitting of the curve in the graph (below).

    [<img src="https://miro.medium.com/max/1400/1*i_lp_hUFyUD_Sq4pLer28g.png" width="50%" />](https://miro.medium.com/max/1400/1*i_lp_hUFyUD_Sq4pLer28g.png)

    As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve.

    **`So, what is the right numbers of epochs?`**

    Unfortunately, there is no right answer to this question. The answer is different for different datasets but you can say that the numbers of epochs is related to how diverse your data is...

* ### **Batch Size**

      Total number of training examples present in a single batch.
    
    > Note: Batch size and number of batches are two different things.
    
    **`But What is a Batch?`** <br/>

    As I said, you can’t pass the entire dataset into the neural net at once. So, you divide dataset into Number of Batches or sets or parts.
    Just like you divide a big article into multiple sets/batches/parts

* ### **Iterations**

      Iterations is the number of batches needed to complete one epoch.

   > **Note**: The number of batches is equal to number of iterations for one epoch.

    Let’s say we have 2000 training examples that we are going to use .
    We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.<br/>

    **Where Batch Size is 500 and Iterations is 4, for 1 complete epoch.**

* ### **Learning rate**

    **`Gradient Descent`** <br/>
    It is an iterative optimization algorithm used in machine learning to find the best results (minima of a curve).<br/>
    Gradient means the rate of inclination or declination of a slope.<br/>
    Descent means the instance of descending.<br/>

    The algorithm is iterative means that we need to get the results multiple times to get the most optimal result. The iterative quality of the gradient descent helps a under-fitted graph to make the graph fit optimally to the data.

    [<img src="https://miro.medium.com/max/1400/1*pwPIG-GWHyaPVMVGG5OhAQ.gif" width="70%"/>](https://miro.medium.com/max/1400/1*pwPIG-GWHyaPVMVGG5OhAQ.gif)

    [<img src="https://miro.medium.com/max/1276/0*FA9UmDXdzYzuOpeO.jpg" width="40%"/>](https://miro.medium.com/max/1276/0*FA9UmDXdzYzuOpeO.jpg)

  The Gradient descent has a parameter called learning rate. As you can see above (left), initially the steps are bigger that means the learning rate is higher and as the point goes down the learning rate becomes more smaller by the shorter size of steps. Also,the Cost Function is decreasing or the cost is decreasing 

  **`What are hyperparameters?`**

    Hyperparameters are the variables which determines the network structure(Eg: Number of Hidden Units) and the variables which determine how the network is trained(Eg: Learning Rate).
    Hyperparameters are set before training(before optimizing the weights and bias).

Reference :

* [https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

* [https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a](https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a)

---
## **`Back propagation`**
---

Back-propagation is the essence of neural net training. It is the practice of fine-tuning the weights of a neural net based on the error rate (i.e. loss) obtained in the previous epoch (i.e. iteration). Proper tuning of the weights ensures lower error rates, making the model reliable by increasing its generalization.

Back-propagation is all about feeding this loss backwards in such a way that we can fine-tune the weights based on which. 

The optimization function (Gradient Descent in our example) will help us find the weights that will — hopefully — yield a smaller loss in the next iteration.

Backpropagation is basically minimize the difference between the 
target values we want to learn and output your network currently producing.

#### **Explanation**: 

Suppose we have training dataset , which consist of input and label (target)
Here 

Desire output is : **t** <br/>
predicted output is : **o** <br/>
Loss function : **(o-t)^2** <br/>

**Step by step explanation** :
* Let us say , we got some output after training neural network
* now we compute a value of loss function i.e. (predicted-target)^2
* We use some optimizer like Gradient descent and SGD to minimize the lose or cost function.   
* now we modify the weight (in the region between output and last hidden layer) to get the 
Target output , with help partial derivative by applying chain rule.
* similary we go one step backword at each step and update the weight. 
* Now again we feed forward a network with new weights , after updating weights
it seems that error or loss is reducing and we reach the point where gradient is zero i.e. Global minima.

---
**During gradient descent** :

  [<img src="https://miro.medium.com/max/1400/1*6sDUTAbKX_ICVVAjunCo3g.png" width="40%"/>](https://miro.medium.com/max/1400/1*6sDUTAbKX_ICVVAjunCo3g.png)

Let’s check the derivative.
- If it is positive, meaning the error increases if we increase the weights, then we should decrease the weight.
- If it’s negative, meaning the error decreases if we increase the weights, then we should increase the weight.
- If it’s 0, we do nothing, we reach our stable point.

**Thus as a general rule of weight updates is the delta rule**:

    w(new) = w(old) - n*(dL/dw)
    New weight = old weight — learning rate * Derivative

  The learning rate is introduced as a constant (usually very small), in order to force the weight to get updated very smoothly and slowly (to avoid big steps and chaotic behaviour). (To remember: Learn slow and steady!)

  In a simple matter, we are designing a process that acts like gravity. No matter where we randomly initialize the ball on this error function curve, there is a kind of force field that drives the ball back to the lowest energy level of ground 0.

  [<img src="https://miro.medium.com/max/1360/1*dvgzK4beVXBGBELDXP9JpA.png" width="40%"/>](https://miro.medium.com/max/1360/1*dvgzK4beVXBGBELDXP9JpA.png)

---

  1. Linear Regression is Neural network , then activation function is identity function. <br/> 
  Loss function is Mean Squared Error Loss or Mean Squared Logarithmic Error Loss or Mean Absolute Error Loss

  2. Logistic Regression is Neural network , then activation function is sigmoid function. <br/>
  Loss function is Binary Cross-Entropy or Hinge Loss or Squared Hinge Loss.

  3. Multiclass classification is Neural network , then activation function is sigmoid function. <br/>
  Loss function is Multi-Class Cross-Entropy Loss or Sparse Multiclass Cross-Entropy Loss or Kullback Leibler Divergence Loss.
    

  **Overall picture**

  In order to summarize, Here is what the learning process on neural networks looks like (A full picture):


[<img src="https://miro.medium.com/max/1400/1*mi-10dMgdMLQbIHkrG6-jQ.png" width="80%"/>](https://miro.medium.com/max/1400/1*mi-10dMgdMLQbIHkrG6-jQ.png)




**Reference** :

* [https://www.youtube.com/watch?v=GJXKOrqZauk](https://www.youtube.com/watch?v=GJXKOrqZauk)

* [http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html](http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)

* [How-does-back-propagation-in-artificial-neural-networks-work](https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7)

* [Neural-networks-and-backpropagation-explained-in-a-simple-way](https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e)

**Reference** :

* [Overview of forward propagation](https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html)

* [https://playground.tensorflow.org](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.58811&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

* [https://openai.com/blog/deep-double-descent/](https://openai.com/blog/deep-double-descent/)

* [http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html](http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)

* [Learning-to-smell-using-deep-learning](https://ai.googleblog.com/2019/10/learning-to-smell-using-deep-learning.html)

* [https://www.youtube.com/watch?v=tIeHLnjs5U8](https://www.youtube.com/watch?v=tIeHLnjs5U8)

* [https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
