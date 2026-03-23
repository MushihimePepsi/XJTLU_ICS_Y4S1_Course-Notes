@[toc]
# 1. 监督学习
监督学习是一种机器学习方法，其中模型从标记的训练数据中学习，以便能够预测新数据的目标变量值。
在监督学习中，除了描述数据的特征变量外，还有一个目标变量（target variable）。
监督学习的目标是学习一个函数（模型），该函数可以根据特征估计或预测目标变量的值。这个函数是通过使用标记的训练集来学习的。
监督学习主要有两大类，根据变量的离散和连续进行区分：
1. 回归（Regression）：目标变量（以及特征）是数值型和连续的。例如，股票价格、国家GDP、班级成绩、儿童身高、预期寿命等。
2. 分类（Classification）：目标变量是离散的。例如，判断纳税人是否作弊、股票是上涨还是下跌、学生是否会通过考试、交易是否欺诈、文章的主题是什么等。

## 1.1 建模方法
在数据分析中主要有两种建模方法：
1. 描述性建模（Descriptive modeling）
描述性建模是一种解释性工具，用于理解数据。它帮助我们分析和解释数据中的模式和关系。
- 回归（Regression）：分析不同因素的变化如何影响目标变量。例如：
哪些因素影响股票价格？
哪些因素影响一个国家的GDP？
- 分类（Classification）：理解不同类别对象之间的区分属性。例如：
为什么人们会在税务上作弊？
哪些词汇会使帖子具有攻击性？
2. 预测性建模（Predictive modeling）
预测性建模用于预测以前未见过的记录的类别。
- 回归（Regression）：预测连续变量的值。例如：
预测一个患者的预期寿命是多少？
- 分类（Classification）：预测离散类别。例如：
判断一个人是否是骗子？
预测股票是否会上涨？
判断一个帖子是否具有攻击性？

## 1.2 监督学习的流程
下图展示了监督学习的基本流程。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/40aaf0be4cb948f0a94e508f2d68d0a0.png)
数据先分为训练数据和测试数据。训练数据用于训练模型，测试数据用于评估模型的性能。
然后我们使用训练数据来训练模型。通过学习算法，模型能够学习特征向量和目标变量之间的关系。
然后我们得到了模型，这个模型可以用刚刚的测试数据来评估模型的性能。通过比较模型预测的目标变量值和实际的目标变量值，可以评估模型的准确性。
我们还可以将训练好的模型应用于新的数据从而开始进行预测。
这里模型是一个函数$M$，它接受特征向量$f$作为输入，并输出目标变量$t$的值。
# 2. 回归
我们先介绍回归。回归分析是统计学和机器学习中的一种预测建模方法，用于估计变量之间的关系。
假设我们有$k$个特征变量，这些变量是数值型的。这也被称为协变量（covariates）或自变量（independent variables）。
目标变量（Target variable）也称为因变量（dependent variable）。
给定一个形式为$\{(x_1,y_1),...,(x_n,y_n)\}$ 的数据集，其中$x_i$是一个$k$维的特征向量，$y_i$是一个实数值。
我们希望学习一个函数$f$，给定一个特征向量$x_i$，该函数能够预测一个值$y ′=f(x_i)$，这个值尽可能接近实际的$y_i$值。
我们的目标是最小化预测值和实际值之间的平方差之和，即：$\sum_{i} (y_i - f(x_i))^2$
这个表达式是回归分析中常用的损失函数，称为均方误差（Mean Squared Error, MSE）。

## 2.1 线性回归
在线性回归中，函数$f$通常具有以下形式：$f(x_i) = w_0 + \sum_{j=1}^{k} w_j x_{ij}$其中，$w_0$是截距项，$w_j$是第$j$个特征变量的权重，$x_{ij}$ 是第$i$个样本的第$j$个特征变量的值。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba0aa28459ea4b7ea04f4dd886fac7c6.png)
### 2.1.1 一元线性回归（One-dimensional linear regression）
在最简单的情况下，我们有一个单一变量，函数形式为：$f(x_i) = w_0 + w_1 x_i$。
通过最小化误差得到的参数计算公式为：
$w_0 = \bar{y} - w_1 \bar{x}$，
$w_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} = r_{xy} \frac{\sigma_y}{\sigma_x}$，
其中，$\bar{x}$和$\bar{y}$分别是$x_i$和$y_i$的均值。
$r_{xy}$是$x$和$y$之间的相关系数。
$σ_x$和$σ_y$分别是$x_i$和$y_i$的标准差。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba0aa28459ea4b7ea04f4dd886fac7c6.png)
### 2.1.2 多元线性回归（Multiple Linear Regression）
多元线性回归用于处理多个特征变量的情况。
在一般情况下，我们有$k$个特征变量，$x_i$和$w$都是向量。表示如下：
$x_i=(1,x_{i1},...,x_{ik})$：每个样本的特征向量，第一个元素是1，用于表示截距项。
$w = (w_0, w_1, ..., w_k)$：权重向量，$w_0$是截距项的权重，其他是特征变量的权重。
$f(x_i, w) = x_i^T w$：预测函数，表示特征向量$x_i$和权重向量$w$的点积。
让$X$表示$n \times (k+1)$的矩阵，其中$x_i$作为行。
让$y = (y_1, ..., y_n)$表示目标变量的向量。
SSE（Sum of Squared Errors）函数可以写为：
$SSE = \|Xw - y\|^2$，这是预测值和实际值之间差的平方和。
对于权重向量$w$，存在一个闭式解：$w = (X^T X)^{-1} X^T y$这个公式通过最小化SSE来求解权重向量。
矩阵求逆可能计算成本很高。其他优化技术（如梯度下降）通常用于找到最优向量。

## 2.2 异常值（Outliers）
回归模型对异常值非常敏感。
下图展示了包含异常值的回归线，可以看到回归线明显受到异常值的影响，向这些极端值倾斜。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c7f6b149eb1349648f567621ca982f3e.png)

一种常见的处理异常值的方法是将它们从数据集中移除。
下图展示了移除异常值后的回归线，可以看到回归线更加平滑，更好地拟合了大部分数据点。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8ed0cfa446d44ec9474bb52cac6b9bf.png)
但是要注意在移除异常值之前，需要确保这些异常值不包含有用的信息。有时候，异常值可能代表了重要的异常情况或趋势，简单地移除它们可能会导致信息的丢失。

## 2.3 标准化（Normalization）
在回归问题中，有时我们的特征变量可能具有非常不同的尺度（scales）。
例如，如果我们使用一个国家的房主比例和收入作为特征来预测GDP，这两个特征的数值范围可能相差很大，导致模型的权重不可解释。
为了解决特征尺度不同的问题，可以通过将特征值替换为z分数（z-scores）来进行标准化。
z分数是通过移除特征的均值（mean）并除以其标准差（standard deviation）来计算的。公式如下：$z = \frac{(x - \mu)}{\sigma}$，其中，$x$是原始特征值，$μ$是特征的均值，$σ$是特征的标准差。
标准化的好处有两点：
1. 标准化后的特征具有均值为0和标准差为1的性质，这有助于改善模型的收敛速度和性能。
2. 标准化后的特征权重更容易解释，因为它们都在相同的尺度上。

## 2.4 更复杂模型
我们使用的模型在参数$w$方面是线性的，但我们考虑的特征可能是$x_i$值的非线性函数。
为了捕捉更复杂的关系，我们可以对输入数据进行转换（例如，取对数$logx_{ij}$	​），或者添加多项式项（例如，$x_{ij}^2$)。
例如，我们可以学习一个形式为$f(x)=w_0+w_1x+w_2x^2$的函数。
然而，这种方法可能会大大增加特征的数量。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28dfff5753c0482baa283af6e3efa9c7.png)
## 2.5 回归模型中系数的解释和显著性检验
回归模型不仅对新数据的预测有用，而且其系数对于理解自变量对因变量的影响也很有用。
$w_j$值表示当$x_{ij}$增加一个单位时，$y_i$的变化量。
可以通过检验$w_j$的显著性来确定其值是否显著不为零。这通常通过检验零假设$H_0:w_j=0$来完成。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e1e52e5bbb644f248ed8056ba908b89b.png)
表格展示了一个多元回归程序的典型输出，包括每个协变量的最小二乘估计值、标准误差、估计的t值和p值。
星号（*）表示显著性水平，更多的星号表示在更小的显著性水平上显著。

通过这些检验，我们可以确定哪些自变量对因变量有显著影响，从而更好地理解和预测数据。
# 3. 分类
分类问题与回归问题类似，我们都有特征（features）和一个我们想要建模或预测的目标变量（target variable）。
在分类问题中，目标变量是离散的，通常被称为类别标签（class label）。
在最简单的情况下，目标变量是一个二元变量，意味着它只有两个可能的类别。
在分类中，特征也可能是分类的（categorical），即它们可以是离散的类别，而不仅仅是数值。

下面我们展示一个分类问题的例子，我们尝试利用机器学习模型来识别可能的逃税行为。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bd5290aaf8fd47979a1940db69131b9e.png)
需要判断下图的2024的新报税记录是否是一个逃税的报税记录。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/39ca639b4d8f4ecbb648c4faf8e539e2.png)
这是一个典型的分类问题，目标是学习一种方法来区分不同类别的记录（逃税者与非逃税者）。
我们再具体一点，分类是学习一个目标函数$f$的任务，该函数将属性集$x$映射到预定义的类别标签$y$之一。例如，如果一个人是单身并且收入低于125K美元，则预测为“不逃税”。
在这个例子中，有两个类别标签（或类别）：Yes（1）和No（0），分别代表逃税和不逃税。

## 3.1 分类的基本流程
分类模型的基本流程：输入属性集$x$，通过分类模型，输出类别标签$y$。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8209846c2fd34142834938190da336f8.png)
## 3.2 实际应用
分类的实际应用示例有很多，下面进行举例：
- 预测肿瘤细胞：将肿瘤细胞分类为良性或恶性。这是一个二分类问题，对于医疗诊断和治疗计划至关重要。
- 分类信用卡交易：将信用卡交易分类为合法或欺诈行为。这有助于防止金融欺诈和保护消费者。
- 新闻故事分类：将新闻故事分类为不同的类别，如财经、天气、娱乐、体育等。这有助于内容组织和推荐系统。
- 识别垃圾邮件和不当内容：识别并分类垃圾邮件、垃圾网页和成人内容。这对于电子邮件过滤和内容监管非常重要。
- 理解网络查询意图：判断一个网络查询是否具有商业意图。这对于在线广告和市场营销策略很有用。

分类在数据科学中无处不在，通过分析和分类大量数据，我们可以发现模式、做出预测并解决各种问题。

## 3.3 分类任务的一般方法和步骤
我们解决分类任务的步骤如下：
1. 获取训练集：
首先，需要获取一个包含已知类别标签的记录的训练集。这些数据用于训练模型，使其能够学习不同特征与类别标签之间的关系。
2. 构建分类模型：
使用训练集来构建或训练一个分类模型。这个过程涉及选择适当的算法，并调整模型参数以最佳地拟合训练数据。
3. 评估模型质量：
使用一个之前未见过的、带有标签的测试集来评估模型的质量。测试集用于验证模型在新数据上的泛化能力，即模型对未知数据的预测能力。
4. 应用分类模型：
将训练好的分类模型应用于带有未知类别标签的新记录。这是模型的实际应用阶段，用于对新数据进行分类预测。
5. 选择特征：
在整个过程中，一个重要的中间步骤是决定使用哪些特征。特征选择对于模型的性能至关重要，因为它影响模型的准确性和效率。

下图展示了这个过程。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b46271347f5946588c508c42cf02203d.png)
通过归纳（Induction）过程，从训练集中学习一个模型，这个模型是通过学习算法从训练集中得到的。我们最后使用训练好的模型对测试集中的新数据进行预测。

## 3.4 分类模型的评估
我们可以使用混淆矩阵（Confusion Matrix）来计算准确率（Accuracy）和错误率（Error rate）。

混淆矩阵是一个表格，用于显示分类模型的预测结果与实际结果的对比。
矩阵的行表示实际的类别（Actual Class），列表示预测的类别（Predicted Class）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5688bf43b2fe4e69aca04925ae4fc7fb.png)
$f _{11}$：真正例（True Positives, TP），模型正确预测为类别1的数量。
$f_{10}$：假负例（False Negatives, FN），模型错误预测为类别0但实际为类别1的数量。
$f _{01}$：假正例（False Positives, FP），模型错误预测为类别1但实际为类别0的数量。
$f_{00}$：真负例（True Negatives, TN），模型正确预测为类别0的数量。
我们记忆的时候可以将其记为：如果结果是正确的就是真，否则为假，正负说的是预测结果，预测是1就是正，否则就是负。
准确率（Accuracy）的计算公式为：
$\text{Accuracy} = \frac{f_{11} + f_{00}}{f_{11} + f_{10} + f_{01} + f_{00}}$
错误率（Error rate）的计算公式为：
$\text{Error rate} = \frac{f_{10} + f_{01}}{f_{11} + f_{10} + f_{01} + f_{00}}$

## 3.5 常用的分类技术
常用的分类技术如下：
1. 基于决策树的方法（Decision Tree based Methods）：
使用树状模型来做出决策，通过一系列的规则对数据进行分类。
2. 基于规则的方法（Rule-based Methods）：
根据一组预定义的规则来对数据进行分类。
3. 基于记忆的推理（Memory based reasoning）：
利用过去的经验或记忆来进行推理和分类。
4. 神经网络（Neural Networks）：
模仿人脑神经元网络的结构来进行学习和分类。
5. 朴素贝叶斯和贝叶斯信念网络（Naïve Bayes and Bayesian Belief Networks）：
基于概率理论的分类方法，其中朴素贝叶斯假设特征之间相互独立。
6. 支持向量机（Support Vector Machines, SVM）：
通过找到最优的决策边界来最大化不同类别之间的间隔。
7. 逻辑回归（Logistic Regression）：
一种预测二分类结果的回归分析方法。

数据分析中最难的事情之一就是知道应该投入时间学习哪种工具来最好地解决一个特定的任务。

# 4. 决策树
我们下面详细介绍决策树算法。
决策树是一种类似流程图的树形结构。
其中内部节点表示对某个属性的测试。这些节点根据特征的不同值将数据分割成不同的分支。
分支代表测试的结果，即根据内部节点的测试，数据将沿着哪个路径继续前进。
叶节点代表最终的分类结果或类别分布。在分类问题中，叶节点通常包含类别标签；在回归问题中，叶节点可能包含预测的数值。

我们回到前面的例子，我们可以给出这样的决策树以判断是否逃税。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43eb092c9a574c2bb8023650dfc0b65e.png)
如果一个样本的“退税”属性为“是”，则直接预测为“逃税”；如果“退税”属性为“否”，则进一步根据“婚姻状况”和“应税收入”属性进行分类。

相同的数据集也可以存在多个决策树模型，如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bce1381d9adc41a0a82943eee7c41a4f.png)
如果一个人已婚，则直接预测为“不逃税”；如果是单身或离婚，则进一步检查是否有退税，以及应税收入是否超过80K。

当我们有这样的决策树后，我们可以应用数据以预测其结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bb78f0618ba34b8d8d541b9af083177a.png)
从决策树的根节点开始，根据测试数据的属性值沿着决策树的分支向下移动。
测试数据的“退税状态”为“No”，因此沿着“No”分支向下移动到“Marital Status”节点。
由于“婚姻状况”为“Married”，继续沿着“Married”分支向下移动。
最后，到达叶节点，根据决策树的规则，预测“是否逃税”为“No”。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b6942d73d7114b69a0cccfc66e48c106.png)

所以在这个分类任务中，我们的学习算法是树归纳算法（Tree Induction Algorithm），学习模型是决策树（Decision Tree）。

## 4.1 树归纳（Tree Induction）
我们现在研究如何找到在训练数据上具有低分类错误的树（即训练错误）。
找到最佳决策树（具有最低训练错误）是一个NP-hard问题。
我们可以尝试使用贪婪策略（Greedy Strategy）作为一种启发式方法，它在每一步选择当前最优的决策，而不考虑未来的结果。
当然还有其他构建决策树的算法：
- Hunt's Algorithm：最早的决策树算法之一。
- CART（Classification and Regression Trees）：一种流行的决策树算法，可以用于分类和回归任务。
- ID3, C4.5：基于信息增益的决策树算法，C4.5是ID3的改进版本，可以处理连续属性。
- SLIQ, SPRINT：其他决策树算法，用于提高构建和预测的效率。

### 4.1.1 亨特算法（Hunt's Algorithm）
$D_t$表示到达节点$t$的训练记录集。
一般过程如下：
- 如果$D_t$包含属于同一类别$y_t$的记录，则$t$是一个标记为$y_t$的叶节点。
- 如果$D_t$包含具有相同属性值的记录，则$t$是一个标记为多数类$y_t$的叶节点。
- 如果$D_t$是一个空集，则$t$是一个标记为默认类$y_d$的叶节点。
- 如果$D_t$包含属于多个类别的记录，则使用属性测试将数据分割成更小的子集。

我们对每个子集递归地应用上述过程。
我们回到前面的例子。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2fd31906e99c436b984104e5b327e1d9.png)
我们现在包含多个类别的记录，我们使用退税属性进行分割，然后可以得到一个值含不逃税的类别和一个包含多个类别的记录，前者我们可以标记为一个叶节点。
后者我们再使用婚姻状况属性进行分隔，同样这里我们可以得到一个叶节点和一个包含多个类别的记录。
我们再使用应税收入属性进行分隔，即可得到两个叶节点，如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e0ecd6d51cc407f9a9e75487f51e901.png)
### 4.1.2 构建决策树的算法
下面是构建决策树的伪代码。
```
GenDecTree(Sample S, Features F)
If stopping_condition(S,F) = true then
   leaf = createNode()
   leaf.label= Classify(S)
   return leaf
root = createNode()
root.test_condition = findBestSplit(S,F)
V = {v | v a possible outcome of root.test_condition}
for each value v ∈ V:
   S_v: = {s | root.test_condition(s) = v and s ∈ S};
   child = GenDecTree(S_v ,F);
   Add child as a descent of root and label the edge (root->child) as v
return root
```
GenDecTree(Sample S, Features F)：这是一个生成决策树的函数，其中 S 是样本集，F 是特征集。
如果满足停止条件（stopping_condition(S,F) = true），则创建一个叶节点（leaf = createNode()），并根据样本集 S 对其分类（leaf.label = Classify(S)），然后返回该叶节点。
创建一个新的节点作为树的根节点（root = createNode()）。
找到最佳分割条件（root.test_condition = findBestSplit(S,F)），这通常是基于某种标准（如信息增益、基尼系数）来选择最佳特征和分割点。
V 是根节点测试条件所有可能结果的集合。
对于 V 中的每个值 v，根据 v 分割样本集 S（S_v := {s | root.test_condition(s) = v and s ∈ S}），递归地为每个子集 S_v 生成子树（child = GenDecTree(S_v, F)），并将子树添加为根节点的后代，同时标记边（root->child）为 v。
最后返回构建好的决策树的根节点。

### 4.1.3 关键问题
我们刚刚只谈到了算法的步骤，但是有一些关键问题并没有给出详细的解答。
1. 如何分类叶节点？
给叶节点分配多数类（Assign the majority class）：如果叶节点包含多个类别的记录，通常将叶节点分类为该节点中出现次数最多的类别。
如果叶节点为空（即没有记录），则分配默认类（default class）：这通常是整个数据集中或其父节点中出现频率最高的类别。
2. 如何确定分割记录？
这个问题上我们需要确定基于哪个属性以及该属性的哪个值来分割数据。需要选择一个属性和分割点，使得分割后的数据集在某种意义上（如信息增益、基尼系数等）更加纯净。
这个问题可以被理解为如何指定测试条件，我们将在稍后详细讨论。
3. 如何确定何时停止分割？
需要确定一个停止条件，以避免过拟合。这可以是基于达到某个树的深度、节点中的记录数少于某个阈值、或者分割后没有明显的性能提升等。

#### 4.1.3.1 如何指定测试条件
- 我们可以根据属性类型进行分隔。
名义属性（Nominal）：属性的值是离散的，没有顺序关系，例如颜色（红、绿、蓝）。
序数属性（Ordinal）：属性的值是有序的，但数值之间的差异可能不代表相同的间隔，例如教育水平（小学、中学、大学）。
连续属性（Continuous）：属性的值可以在一个范围内连续变化，例如温度或收入。
- 也可依赖于分割方式的数量进行分隔。
二元分割（2-way split）：属性被分割成两个分支，例如，对于一个连续属性，可以选择一个阈值进行分割，如“收入 > 50000”。
多元分割（Multi-way split）：属性被分割成多个分支，这通常用于名义属性，因为名义属性可以有多个可能的值。

现在详细讲述每种情况：
1. 名义属性（Nominal）：
多元分割（Multi-way split）：
使用与属性的不同值一样多的分割。例如，如果一个属性有三个不同的值，那么多元分割会创建三个分割。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/500cabda9aba4e8b9991825df5610f06.png)
二元分割（Binary split）：
将值分成两个子集。这通常涉及到选择一个属性的两个值来创建两个分割。
需要找到最优的分割方式，以最大化分割后子集的纯度（例如，通过信息增益或基尼系数）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a788e832bd548fab17ff6caf96619ce.png)
2. 序数属性（Ordinal）：
多元分割（Multi-way split）：这里“Size”属性有三个值：Small、Medium 和 Large，因此创建了三个分割。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a70360534bac4236bef5f169ba201dcb.png)
二元分割（Binary split）：
这里展示两种可能的方案。一种是将“Size”分割为“{Small, Medium}”和“{Large}”。另一种是将“Size”分割为“{Medium, Large}”和“{Small}”。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3548a03d89c7449eb392b96898d83d5e.png)
那我们能不能将“Size”分割为“{Small, Large}”和“{Medium}”呢？。这种分割很可能不是最优的，因为它没有尊重属性值的顺序，可能会导致信息的损失。

3. 连续属性（Continuous）：
连续属性的处理比较复杂，因为连续属性是指可以在某个范围内取任何值的属性，例如年龄、收入或温度。
我们需要将连续属性转换为序数（ordinal）分类属性。有两种方式：
- 静态（Static）：在开始时一次性进行离散化。
- 动态（Dynamic）：可以通过等间隔分桶（equal interval bucketing）、等频率分桶（equal frequency bucketing，即百分位）或聚类（clustering）来找到范围。

我们也可以使用二元决策去处理：
通过比较属性值与某个阈值（v）来创建二元分割，即$A<v$或$A≥v$。
需要考虑所有可能的分割点，并找到最佳分割点。这种方法可能在计算上更为密集。

多元分割（Multi-way split）：这里将“应税收入”分割为多个区间：< 10K, [10K, 25K), [25K, 50K), [50K, 80K), > 80K。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4103ff0451594cdb9d02ee434e018211.png)
二元分割（Binary split）：这里将“应税收入”（Taxable Income）是否大于80K作为分割条件。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/760f59effb7e41db86c2fbd5cfbfd5d9.png)

#### 4.1.3.2 最佳分割
我们刚刚一直说到最佳分割点，那到底什么是好的分割，什么是差的分割呢？
假设我们有一组数据有10个class0和10个class1，下面是几种分割方案。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3680f92b40de4b8ab848369b57495d93.png)
我们现在尝试找到一种方式去评估这些分割策略。
我们使用节点不纯度（impurity）来度量。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/907e6bbfca1440b9b25b5fa27a59b7c6.png)
左边的节点包含5条类别0（C0）的记录和5条类别1（C1）的记录，这是一个非同质的（non-homogeneous）节点，具有高度的不纯度（high degree of impurity）。
右边的节点包含9条类别0的记录和1条类别1的记录，这是一个同质的（homogeneous）节点，具有低度的不纯度（low degree of impurity）。

##### 4.1.3.2.1 节点不纯度（Node impurity）
我们使用节点$D_t$表示决策树中的一个节点，该节点包含的样本属于多个类别$\{1,...,c\}$。
$p(i∣t)$：表示在节点$D_t$中属于类别$i$的记录的比例。
我们使用熵（Entropy）来进行系统的度量，其的计算如下：$Entropy(D_t) = - \sum_{i=1}^{c} p(i|t) \log p(i|t)$
熵用于ID3和C4.5算法中，它衡量节点中样本类别分布的混乱程度，值越高表示不纯度越高。
我们还可以使用基尼系数（Gini）来度量不纯度，其的计算如下：
$Gini(D_t) = 1 - \sum_{i=1}^{c} p(i|t)^2$
用于CART、SLIQ、SPRINT算法中，它也是衡量不纯度的指标，值越低表示节点越纯。
我们还可以使用分类错误率去度量，计算如下：
$Classification Error (D_t) = 1 - max p(i|t)$
表示节点中样本最多的类别所占的比例，值越低表示纯度越高，也就是不纯度越低。

我们可以将这三个都应用于前面的例子中。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d41a903eab5d4648a1fa1c4cc78896c8.png)

我们都可以得到方案2的分割更好。
##### 4.1.3.2.2 Python中应用
我们可以使用Python的scikit-learn库中的DecisionTreeClassifier类来创建一个决策树分类器。

```python
class sklearn.tree.DecisionTreeClassifier(
    *, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
    max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None,
    ccp_alpha=0.0, monotonic_cst=None)  # 
```
criterion参数：用于指定衡量分割质量的函数。可选的值包括"gini"（基尼系数）、"entropy"（信息熵）和"log_loss"（对数损失），默认值为"gini"。
max_depth参数：用于设置树的最大深度，即树中从根节点到叶节点的最长路径上的节点数。默认值为None，表示树的深度不受限制。
max_features参数：用于指定在寻找最佳分割时要考虑的特征数量。可以是整数、浮点数或{"sqrt", "log2"}中的一个。默认值为None，表示使用所有特征。
splitter：指定分割策略，'best'表示选择最佳分割。
min_samples_split：指定进行分割所需的最小样本数。
min_samples_leaf：指定叶节点所需的最小样本数。
min_weight_fraction_leaf：指定叶节点的最小样本权重比例。
random_state：控制随机数生成器的状态。
max_leaf_nodes：指定最大叶节点数。
min_impurity_decrease：指定节点不纯度的最小增加量。
class_weight：指定类别权重。
ccp_alpha：复杂度剪枝剪切罚系数的参数。
monotonic_cst：指定是否为单调递增。

下面给出具体的代码示例。

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifier
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_split=5,
    random_state=42
)

# Train and predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Optional: Visualize the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

#### 4.1.3.3 解释决策树
我们不仅要学会使用决策树进行分类，我们还需学会如何结束决策树。
以下图为例子。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49326696844042f1bd08f2a116511e92.png)
根节点是决策树的起始点，它基于最重要的特征进行第一次分割。这里根据“Outlook”属性，将数据分割为“Sunny”和“Rainy”或“Overcast”。
基尼系数是一个介于0和0.5之间的值，值越接近0表示节点越纯净（即样本属于同一类别），越接近0.5表示节点越混乱（即样本属于不同类别）。
样本（Samples）是再该分支上的总样本数。
值表示每个类别的样本比例，这里，[No=3, Yes=6] 表示3个样本属于“No”类别，6个样本属于“Yes”类别。
类别（Class）是在该分支上预测的多数类，如果分割在这里停止，则该节点为叶节点。