## Lec 7
1. **写出复合命题 $(p \lor \neg q) \rightarrow (p \land q)$ 的真值表。**

    | $p$ | $q$ | $\neg q$ | $p \lor \neg q$ | $p \land q$ | **$(p \lor \neg q) \rightarrow (p \land q)$** |
    |:---:|:---:|:-------:|:------------:|:---------:|:--------------------------------------:|
    |  T  |  T  |    F    |      T       |     T     |                  **T**                     |
    |  T  |  F  |    T    |      T       |     F     |                  **F**                     |
    |  F  |  T  |    F    |      F       |     F     |                  **T**                     |
    |  F  |  F  |    T    |      T       |     F     |                  **F**                     |

    

2. **给定知识库中的句子：**
   
    -  $P \rightarrow Q$ 
    -  $Q \rightarrow R$
    -  $P$
    
    使用 Modus Ponens 推理规则证明 $R$ 。
    
    
    
    ​	我们知道，Modus Ponens（假言推理规则）的形式是：
    
    $$
    \frac{\quad P, P \rightarrow Q}{\therefore Q}
    $$
    
    ​	现在，运用 Modus Ponens 来推导 $R$：
    
    1. 从 $P \rightarrow Q$ 和 $P$ 推导出 $Q$：
    $$
    \frac{\quad P, P \rightarrow Q}{\therefore Q}
    $$
    
    2. 从 $Q \rightarrow R$ 和 $Q$ 推导出 $R$：
    $$
    \frac{\quad Q, Q \rightarrow R}{\therefore R}
    $$
    
    ​	到此，我们已经证明了$R$.
    

---

## Lec 8

1. **全称量词和存在量词的应用**

    使用一阶逻辑的全称量词和存在量词来表达以下两个句子：

    (1) 所有的国王都是富有的。

    ​	$$ \forall x \ King(x) \Rightarrow Rich(x) $$

    

    (2) 有些国王是富有的。

    ​	$$\exists x \ King(x) \wedge Rich(x) $$
    
    


2. **等词应用**

    等词用于表达两个项指代同一对象或不同对象（加否定词时表示两个项不是同一个对象）：
    (1) Richard 至少有两个兄弟。

    ​	$$\exists x,y \ Brother(x, \ Richard) \wedge Brother(y, \ Richard) \wedge \neg (x = y)$$

    

    (2) Richard 有两个兄弟 x 和 y。

    ​	$$ Brother(x, \ Richard) \wedge Brother(y, \ Richard) \wedge (x = y) \wedge \forall i \ Brother(i, \ Richard ) \Rightarrow (i=x \vee i=y) $$

    

3. **推理**

    使用前向链接和后向链接，从已知事实推导出新事实：

    (1) 已知知识库中有以下信息：
    - $\forall x \ \text{King}(x) \Rightarrow \text{Rich}(x)$ （所有国王都是富有的）
    - $\text{King(Charles)}$ （查尔斯是国王）
      
    
    问题：使用前向链接推理出查尔斯是富有的。
    
    ​	首先消去全称量词，得 $King(x) \wedge Rich(x)$ ；
    
    ​	再由已知 $King(Charles)$ 得 $King(Charles) \wedge Rich(Charles)$ ，也即 $Rich(Charles)$ ；
    
    ​	所以我们证明了 *Charles* 是富有的。
    
    
    
    (2) 在 A 国家，任何违反环境保护法的行为都被视为犯罪行为。未经授权倾倒有毒废物至环境中是违法的。如果企业能证明其行为是为了防止更大的环境灾害，可以申请倾倒有毒废物的紧急授权。
    
    某湖泊被政府指定为自然保护区。企业家 E 在该湖泊中倾倒了有毒废物。E 声称其行为是为了防止更严重的环境灾害。此外，没有证据直接表明 E 申请了紧急授权。
    
    问题：使用逻辑推理，分析 E 是否犯了罪。
    
    
    
     ​	**首先**，我们可以将这些已知的内容转化为一些一阶确定子句：
	
	
	- 违反环境保护法的行为是犯罪行为：$AgainstLaw(x) \ \Rightarrow Criminal(x)$
	
	- 倾倒有毒废物至环境中：$PourPoison(x, \ env) \wedge Environment(env)$
	
	- 湖泊属于环境：$Lake(L) \Rightarrow Environment(L)$
    
    - 未经授权倾倒有毒废物是违法的：$ PourPoison(x, \ env) \wedge \neg EmergencyAuth(x) \Rightarrow AgainstLaw(x)$
    
    - 声称为了防止更严重的环境灾害：$Claim(x)$
    
    - 证明为了防止更严重的环境灾害：$Prove(x)$
    
    - 证明后，企业可以申请紧急授权来倾倒有毒废物以防止更大的环境灾害，不能证明则无效：
    
    
      - $Prove(x) \Rightarrow EmergencyAuth(x)$
      - $\neg Prove(x) \Rightarrow \neg EmergencyAuth(x)$
    
      
    
      **接下来**，我们可以使用后向链接推理来分析 E 是否犯了罪，假定该湖泊名称为 L，我们已知 $PourPoison(E,L)$ ,$Claim(E)$, $\neg Prove(E)$。
    
      我们的**目标结论**：企业家 E 犯罪了，即 $Criminal(E)$。
    
      为了使用后向链接进行推理，要先逐步确定达成这个结论所需的必要前提条件，然后检查这些条件是否被满足。
    
      1. **必要条件1**：在 A 国，违反环境保护法的行为是犯罪，即要证明 $AgainstLaw(E)$ ；
    
      2. **必要条件2**：E 实施了倒入有毒废物进入湖泊 L 的行为，即 $PourPoison(E,L)$；
    
      3. **必要条件3**：E 未获得紧急授权，即 $\neg EmergencyAuth(E)$；
    
      4. **必要条件4**：湖泊 L 属于环境的范畴，即$Environment(L)$。
    
		证明树如下所示：
    
         ![证明树](https://raw.githubusercontent.com/Jinbao2333/AIFundamentals2024/9249154ea44aa5d6507beb175f3341b0f88216a4/Assignment2/B_Chaining.svg)
    
      现在，我们逆向验证这些条件：
    
      - 从结论 $Criminal(E)$ 出发；
    
      - 我们检查 E 的行为是否违反了环境保护法 $AgainstLaw(E)$ ，这需要 E 进行非法倾倒有毒废物的行为 $PourPoison(E,L)$，并且没有证据获取紧急授权 $\neg EmergencyAuth(E)$；
    
      - 对于第一项条件，E 确实倾倒了有毒废物进入湖泊，湖泊属于环境，也即 $PourPoison(E,L) \wedge Environment(L)$ ；
    
      - 接着，我们验证第二项条件。由于没有证据表明 E 有紧急授权，这一条件不成立，意味着没有合法理由豁免 E 的行为。而且通过 E 的声明 $Claim(E)$ 且没有证据证明其为了避免更大危害，即  $\neg Prove(E)$，也不足以获取紧急授权，所以没有例外情况，我们证明了 $\neg EmergencyAuth(E)$。
    

​		**综上**，以上条件均能满足，我们证明了 E 确实犯罪了，即 $Criminal(E)$。

---

## Lec 9
1. **语义网络**

    用语义网络表示下列信息：

    (1) 胡途是思源公司的经理，他 35 岁，住在飞天胡同 68 号。
    
    (2) 华东师范大学有两个校区：中北校区和闵行校区。
     - 中北校区设有计算机学院，其中张智是计算机科学与技术专业的教授，同时负责人工智能实验室。
     - 闵行校区设有物理学院，李物是量子物理专业的副教授，同时是量子信息研究中心成员。
     - 张智教授最近发表了一篇关于机器学习的研究论文，在人工智能领域获得了广泛认可。
    
    问题：

    (a) 给出华东师范大学在语义网络中的表示。

    (b) 说明张智教授和李物副教授在语义网络中的位置及其关联属性。


2. **用概率来量化不确定性**

    假设你是应急管理部门的一名决策分析师，任务是为城市近期可能发生的自然灾害（例如洪水或地震）制定一个应急响应计划。

    (1) 描述如何使用概率理论来估计接下来一年内城市发生大规模洪水的风险。

    (2) 如果已知城市的不同区域对洪水的脆弱性不同，解释如何通过条件概率来评估特定区域受灾的可能性。

    (3) 考虑多个因素（如降雨量、河流水位等）可能影响洪水发生的概率，如何构建一个概率模型来为应急响应计划提供决策支持。
