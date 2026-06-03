# ReLaHG-LLM：面向跨领域异构图分类的关系文本参数化 HGNN 与标签感知 LLM 预测框架

## 1. 研究问题与新定位

本文研究的问题不再是“严格无标签的 zero-shot learning”，而是：

> **在源领域具有监督信息、目标领域标签稀缺或不参与训练，且不同领域具有不同节点类型、关系类型与标签空间的条件下，如何结合异构图结构表示与大语言模型语义推理能力，实现跨领域节点分类？**

例如，源领域可以是 IMDB 异构图，其中节点类型为 Movie、Actor、Director，任务标签为 Action、Comedy、Drama；目标领域可以是 DBLP 异构图，其中节点类型为 Author、Paper、Venue、Term，任务标签为 Data Mining、Artificial Intelligence、Database、Information Retrieval。

这一问题具有四个核心挑战：

1. **Schema 不一致**：不同领域的节点类型和边类型完全不同；
2. **结构模式不一致**：电影—演员关系与作者—论文关系无法直接共享固定参数；
3. **标签空间不一致**：源领域分类头无法直接迁移到目标领域；
4. **HGNN 与 LLM 的接口不一致**：HGNN 输出连续结构表示，LLM 更擅长在语言/token 语义空间中作出决策。

因此，本文的核心目标不是构造一个固定的源域分类器，而是：

> **学习如何将异构图结构表示转化为 LLM 真正能够利用的预测证据。**

---

## 2. 现有范式的不足与本文出发点

### 2.1 范式一：LLM-as-enhancer

现有一类工作使用 LLM 生成节点描述、summary、伪标签、prompt 或解释，然后由 GNN/HGNN 使用这些信息进行训练并负责最终预测。

该范式的问题在于：

- LLM 主要在数据增强或伪监督阶段发挥作用；
- HGNN 仍是最终 predictor；
- LLM 并不直接约束 HGNN 的输出是否能被语言模型理解；
- LLM 与 HGNN 之间属于松耦合关系。

本文放弃此前 “LLM 生成 prediction-oriented summary 再监督 HGNN” 的路线，避免模型过度依赖 LLM 伪标签式信息。

### 2.2 范式二：LLM-as-predictor with textualized graphs

另一类方法把邻居、路径、metapath 或子图结构转化为文本序列，再由 LLM 直接预测。

该范式的问题在于：

- 图结构必须被线性序列化，可能损害结构保真性；
- 节点排列顺序改变可能影响 prompt 和预测，弱化图的排列不变性；
- 边类型和高阶局部拓扑只能通过文本间接表达；
- 预测质量依赖 prompt 设计、序列化策略和上下文长度。

### 2.3 范式三：Post-hoc embedding/projector alignment

还有一类方法先单独训练图编码器，再训练 projector，将图 embedding 映射到 LLM/token 空间。

该范式的不足是：

- Projector 可能只学习几何上的表示相似；
- HGNN 输出不一定真正改善 LLM 决策；
- 若最终 LLM 预测损失不能反向更新前端图模块，则这种对齐仍然是事后的、松散的。

### 2.4 本文提出的新范式

本文提出：

> **Prediction-driven Differentiable HGNN–LLM Alignment：将冻结 LLM 纳入训练闭环，LLM 作为最终预测器和可微语义判断器；其分类损失经由 graph tokens 反向传播到 projector、标签感知证据模块与 HGNN，使图模块学习生成真正有助于 LLM 决策的结构证据。**

总体流程为：

$$
	ext{Heterogeneous Graph}

ightarrow
	ext{Relation-Text Parameterized HGNN}

ightarrow
	ext{Label-Aware Graph Evidence}

ightarrow
	ext{Graph Tokens}

ightarrow
	ext{Frozen LLM Predictor}

ightarrow
	ext{Prediction}
$$

---

## 3. 总体模型框架

建议模型命名为：

> **ReLaHG-LLM: Relation- and Label-aware Heterogeneous Graph-grounded Large Language Model**

模型由四部分构成：

1. **Relation-Text Parameterized HGNN**：利用关系文本动态调制异构消息传递，实现跨 schema 的图结构编码；
2. **Contrastive / Structural Regularization**：保持基础图表示的通用结构语义，避免仅依赖最终 LLM 分类信号；
3. **Label-Aware Graph Evidence Module**：使用候选标签文本作为 query，从图表示中抽取与每个候选标签相关的结构证据；
4. **Frozen LLM Predictor**：接收节点文本、标签描述和 graph tokens，输出最终预测；其损失反向约束前端图模块。

模型中各模块的职责可以概括为：

| 模块 | 核心职责 |
|---|---|
| Relation-Text Parameterized HGNN | 编码可跨 schema 迁移的异构结构表示 |
| 对比学习 / 结构自监督 | 保持图表示的通用性和结构有效性 |
| Label-Aware Graph Evidence | 根据候选标签提取任务相关图证据 |
| Projector | 将图证据投影为 LLM 可接收的连续 graph tokens |
| Frozen LLM | 最终语义预测器，并提供预测驱动的反向监督 |

---

## 4. 模块一：Relation-Text Parameterized HGNN

## 4.1 设计动机

传统 HGNN 往往为每种关系定义独立参数，例如：

$$
m_{u
ightarrow v}^{r}=W_r h_u
$$

在 IMDB 中，模型需要学习：

$$
W_{	ext{movie-actor}}, \quad W_{	ext{movie-director}}
$$

而在 DBLP 中则需要：

$$
W_{	ext{author-paper}}, \quad W_{	ext{paper-venue}}, \quad W_{	ext{paper-term}}
$$

这些参数依赖具体 relation ID 与数据集 schema，因此在不同领域之间难以共享。

如果将异构图完全同构化后使用普通 GNN，虽然能够共享参数，但会削弱：

- 节点类型差异；
- 关系语义差异；
- 不同类型邻居对目标节点的不同作用；
- 高阶异构结构信息。

因此，本文保留 HGNN 的关系感知消息传递形式，但将传统的固定 relation-specific 参数改为：

> **由关系自然语言描述动态生成的 relation-conditioned 参数。**

模型学习的不是固定关系 ID 对应的参数，而是：

$$
	ext{关系语义}

ightarrow
	ext{消息传递行为}
$$

这使得同一套 HGNN 主干能够在新数据集和新关系类型上复用。

---

## 4.2 节点表示初始化

对于每个节点 $v$，模型使用节点自身文本与节点类型描述构建初始表示。

节点文本编码：

$$
t_v = E_{	ext{text}}(\operatorname{text}(v))
$$

节点类型文本编码：

$$
s_{	au(v)}
=
E_{	ext{text}}(\operatorname{desc}(	au(v)))
$$

例如：

- `Movie`: a film entity with title, actors, directors and keywords.
- `Author`: a researcher who writes academic papers.
- `Venue`: a conference or journal where papers are published.

为避免为每个节点类型定义独立参数，可由节点类型文本生成轻量适配参数：

$$
[\gamma_{	au(v)},eta_{	au(v)}]
=
MLP_{	ext{type}}(s_{	au(v)})
$$

初始节点表示为：

$$
h_v^{(0)}
=
\gamma_{	au(v)}
\odot
MLP_x(t_v)
+
eta_{	au(v)}
$$

其中：

- $MLP_x$ 为共享节点投影网络；
- $\gamma_{	au(v)},eta_{	au(v)}$ 为类型文本动态生成的调制参数。

---

## 4.3 关系文本参数化消息传递

对于边关系 $r$，先构造自然语言描述并编码：

$$
e_r
=
E_{	ext{text}}(\operatorname{desc}(r))
$$

例如：

- `acted_by`: a movie is acted by an actor.
- `directed_by`: a movie is directed by a director.
- `writes`: an author writes a paper.
- `published_in`: a paper is published in a venue.
- `contains_term`: a paper contains a research keyword.

不建议直接让文本生成完整矩阵 $W_r\in\mathbb{R}^{d	imes d}$，因为这种设计参数量大、训练不稳定且可能过拟合。本文更适合采用 FiLM / Adapter 式轻量调制：

$$
[\gamma_r,eta_r]
=
MLP_{	ext{rel}}(e_r)
$$

其中 $MLP_{	ext{rel}}$ 为所有数据集共享的参数生成器。

邻居节点 $u$ 在关系 $r$ 下向节点 $v$ 传递的关系调制表示为：

$$
	ilde{h}_{u}^{r}
=
\gamma_r\odot W_0h_u+eta_r
$$

其中 $W_0$ 是共享消息变换矩阵。

---

## 4.4 关系感知注意力

关系语义不仅决定“传递什么内容”，也应决定“该消息有多重要”。因此，设计关系感知注意力：

$$
a_{uv}^{r}
=
\mathbf{a}^{	op}
	anh
\left(
W_qh_v+
W_kh_u+
W_e e_r

ight)
$$

$$
lpha_{uv}^{r}
=
\operatorname{softmax}_{u\in\mathcal{N}(v)}
\left(
a_{uv}^{r}

ight)
$$

最终消息为：

$$
m_{u
ightarrow v}^{r}
=
lpha_{uv}^{r}
\left(
\gamma_r\odot W_0h_u+eta_r

ight)
$$

节点更新为：

$$
h_v^{(l+1)}
=
\operatorname{LayerNorm}
\left[
h_v^{(l)}
+
\sigma
\left(
\sum_{u\in\mathcal{N}(v)}
m_{u
ightarrow v}^{r}

ight)

ight]
$$

该设计既保留了 HGNN 的异构消息传递能力，又避免模型参数直接绑定某一数据集的 relation ID。

---

## 4.5 哪些内容动态生成，哪些参数共享训练？

| 组成部分 | 是否文本动态生成 | 是否可训练 | 说明 |
|---|---:|---:|---|
| 文本编码器 $E_{	ext{text}}$ | 否 | 通常冻结 | 编码节点、类型、关系和标签文本 |
| 节点共享投影 $MLP_x$ | 否 | 是 | 统一映射节点语义 |
| 类型调制参数 $\gamma_{	au},eta_{	au}$ | 是 | 非独立参数 | 由节点类型文本动态生成 |
| 共享消息矩阵 $W_0$ | 否 | 是 | 提供通用图传播能力 |
| 关系参数生成器 $MLP_{	ext{rel}}$ | 否 | 是 | 学习如何由关系文本生成调制 |
| 关系调制参数 $\gamma_r,eta_r$ | 是 | 非独立参数 | 针对每种关系在 forward 中生成 |
| 注意力参数 $W_q,W_k,W_e,\mathbf{a}$ | 否 | 是 | 所有数据集共享 |
| LayerNorm / 残差结构 | 否 | 是 | 通用图编码结构 |

核心原则是：

> **由文本动态生成传统 HGNN 中与 schema 强绑定的类型/关系调制部分；通用消息传递能力保持共享可训练。**

---

## 5. 模块一的辅助学习：对比学习与结构自监督

## 5.1 为什么加入对比学习？

如果 HGNN 只通过最终 LLM 分类损失训练，图表示可能仅围绕当前预测任务进行调整，而忽视更通用的节点语义与结构信息。在跨领域条件下，这可能造成源领域过拟合。

因此，对比学习的主要作用不是简单地让 HGNN 复制源域类别边界，而是：

- 保持图结构表示与节点自身语义一致；
- 提高图表示稳定性；
- 降低 HGNN 仅依赖下游 LLM 分类梯度的风险；
- 为目标领域迁移提供更通用的结构—语义基础。

---

## 5.2 推荐的节点—文本对比损失

对基础节点表示进行投影：

$$
z_v
=
Proj_{	ext{con}}(h_v)
$$

节点文本表示为：

$$
t_v
=
E_{	ext{text}}(\operatorname{text}(v))
$$

采用 CLIP-style 双向对比学习：

$$
\mathcal{L}_{g
ightarrow t}
=
-rac{1}{|B|}
\sum_{v\in B}
\log
rac{
\exp(\operatorname{sim}(z_v,t_v)/	au)
}{
\sum_{u\in B}
\exp(\operatorname{sim}(z_v,t_u)/	au)
}
$$

$$
\mathcal{L}_{t
ightarrow g}
=
-rac{1}{|B|}
\sum_{v\in B}
\log
rac{
\exp(\operatorname{sim}(t_v,z_v)/	au)
}{
\sum_{u\in B}
\exp(\operatorname{sim}(t_v,z_u)/	au)
}
$$

$$
\mathcal{L}_{NT}
=
rac{1}{2}
\left(
\mathcal{L}_{g
ightarrow t}
+
\mathcal{L}_{t
ightarrow g}

ight)
$$

注意：该损失对齐的是**节点自身文本**，而不是源领域标签描述，因此比直接做节点—标签对齐更有利于跨领域泛化。

---

## 5.3 可选的监督对比学习

源领域有标签时，也可以让同类节点靠近、异类节点分离：

$$
\mathcal{L}_{SupCon}
=
-\sum_{i\in B}
rac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
rac{
\exp(\operatorname{sim}(z_i,z_p)/	au)
}{
\sum_{a\in B,a
eq i}
\exp(\operatorname{sim}(z_i,z_a)/	au)
}
$$

但该损失存在风险：

- 目标领域标签空间可能与源领域完全不同；
- 权重过大可能令图表示围绕源域类别聚集；
- 跨领域迁移能力可能下降。

因此，建议将 $\mathcal{L}_{SupCon}$ 作为可选低权重增强项，而不是模型的核心约束，并通过目标领域性能消融验证其收益。

---

## 5.4 结构自监督损失

为证明模型确实利用了图结构而不是只使用文本，可以加入边或关系重构任务：

$$
\mathcal{L}_{STR}
=
-\log
P(r_{uv}\mid h_u,h_v)
$$

或者：

$$
\mathcal{L}_{STR}
=
-\log
P((u,v)\in E\mid h_u,h_v,e_r)
$$

结构自监督的作用是：

- 强制 HGNN 编码拓扑信息；
- 增强 shuffled-edge 等结构扰动实验中的可验证性；
- 改善图信息较重要场景下的表示质量。

---

## 6. 模块二：Label-Aware Graph Evidence Module

## 6.1 设计动机

Relation-Text Parameterized HGNN 输出的是通用结构表示。但如果仅将一个统一 graph embedding 送入 LLM，会存在三个不足：

1. **证据不具针对性**：通用 graph token 同时包含多种结构语义，未必针对当前候选标签；
2. **目标标签空间变化**：源域与目标域标签不同，模型应根据目标标签语义重新选择图证据；
3. **LLM 决策需要可用证据**：LLM 更需要“为什么该节点符合某候选标签”的结构提示，而不是无差别的图压缩向量。

因此，本文设计 Label-Aware Graph Evidence Module：

> **使用候选标签描述作为语义查询，从 HGNN 生成的子图表示中抽取针对每个候选标签的结构证据。**

---

## 6.2 标签描述编码

对于候选类别 $c$，使用其自然语言描述生成标签语义表示：

$$
e_c
=
E_{	ext{text}}(\operatorname{desc}(c))
$$

例如，在 DBLP 中：

- `Data Mining`: research on discovering patterns and knowledge from large-scale data.
- `Artificial Intelligence`: research on machine learning, reasoning and intelligent systems.
- `Database`: research on data management, query processing and storage systems.

目标领域的候选标签即使在源域训练阶段从未出现，也能够以自然语言描述作为查询入口。

---

## 6.3 以标签语义查询图结构证据

对于待预测中心节点 $v$，HGNN 编码其局部异构子图并得到节点表示集合：

$$
H_{G_v}
=
[h_v,h_{u_1},h_{u_2},\ldots,h_{u_n}]
$$

对于每一个候选标签 $c$，使用标签语义作为 query，子图节点表示作为 key/value：

$$
Q_c=W_Qe_c
$$

$$
K_v=W_KH_{G_v},
\qquad
V_v=W_VH_{G_v}
$$

$$
g_{v,c}
=
\operatorname{softmax}
\left(
rac{Q_cK_v^{	op}}{\sqrt{d}}

ight)
V_v
$$

其中：

$$
g_{v,c}
$$

表示候选标签 $c$ 对应的 label-aware graph evidence。

这一机制允许同一个节点针对不同候选类别关注不同图结构。例如：

- 对 IMDB 节点查询 `Action` 时，模型可能更关注动作类导演、演员共现和相关电影邻居；
- 查询 `Comedy` 时，可能更关注喜剧相关演员或轻松题材邻居；
- 对 DBLP 作者查询 `Data Mining` 时，模型可能更关注 KDD、frequent pattern 和 knowledge discovery 相关论文或术语；
- 查询 `Database` 时，则可能更关注 SIGMOD、VLDB 与 query processing 结构证据。

---

## 6.4 Label-Aware Evidence 与基础 HGNN 表示的职责分离

为了避免损害跨领域泛化，本文不直接强制基础节点表示 $h_v$ 靠近源领域标签 embedding：

$$
h_v 
otpprox e_{y_v}
$$

相反，标签相关语义约束只施加在证据表示上：

$$
g_{v,y_v}
pprox
e_{y_v}
$$

这种职责分离使得：

- $h_v$ 保留通用的结构和节点语义；
- $g_{v,c}$ 承担候选标签相关的证据提取功能；
- 标签对齐不会将基础图空间锁死在源领域类别体系中；
- 目标领域新标签能够通过标签描述动态查询新的结构证据。

---

## 6.5 Graph Token Projection

提取出的标签感知证据需要转化为 LLM 可接收的连续输入：

$$
P_{v,c}
=
Projector(g_{v,c})
$$

其中：

$$
P_{v,c}\in\mathbb{R}^{m	imes d_{	ext{LLM}}}
$$

- $m$ 表示每个候选标签所对应的 graph token 数量；
- $d_{	ext{LLM}}$ 表示 LLM token embedding 维度。

这些 graph tokens 不对应自由生成的文本 summary，而是由异构结构证据形成的连续 soft tokens，直接输入 LLM predictor。

---

## 7. LLM Predictor 与分类损失

## 7.1 LLM 的角色

本文中 LLM 的角色不是：

- 生成 summary 供 HGNN 学习；
- 单独读取序列化图 prompt 完成推理；
- 仅作为冻结文本 embedding encoder。

而是：

> **最终预测器与可微语义判断器。**

HGNN 和 evidence module 的目标是产生能被该 LLM predictor 使用的结构证据。

---

## 7.2 标签评分式预测：Yes/No Verbalizer

不建议让 LLM 在训练时自由生成长文本 explanation。更稳妥的方式是，将分类转化为对每个候选标签的匹配判断。

对节点 $v$ 和候选标签 $c$，构造输入：

$$
\mathcal{I}_{v,c}
=
[
P_{v,c};
\operatorname{text}(v);
\operatorname{desc}(c);
\operatorname{instruction}
]
$$

其中 instruction 为：

> Does the target node belong to this label? Answer Yes or No.

LLM 输出 `Yes` 的 log-probability 作为标签兼容性分数：

$$
s_{v,c}
=
\log
P_{	ext{LLM}}
\left(
	ext{Yes}\mid\mathcal{I}_{v,c}

ight)
$$

对所有候选标签做 softmax：

$$
P(c\mid v)
=
rac{
\exp(s_{v,c})
}{
\sum_{c'\in\mathcal{Y}}
\exp(s_{v,c'})
}
$$

最终分类损失为：

$$
\mathcal{L}_{LLM}
=
-\log
P(y_v\mid v)
$$

这种设计的优点是：

- 输出受控，不需要生成 explanation 监督；
- 每个候选标签均有对应的图证据；
- 能自然适配目标领域的新标签描述；
- 训练梯度能够通过 LLM 回传到前端图模块。

---

## 7.3 梯度传播与参数更新

训练时冻结 LLM backbone，但保留从 LLM 输出到 graph tokens 的梯度路径：

$$
\mathcal{L}_{LLM}

ightarrow
LLM

ightarrow
P_{v,c}

ightarrow
Projector

ightarrow
Label	ext{-}Aware\ Graph\ Evidence

ightarrow
Relation	ext{-}Text\ Parameterized\ HGNN
$$

其中：

$$

abla_{	heta_{	ext{LLM}}}
\mathcal{L}_{LLM}
=
0
$$

而：

$$

abla_{	heta_{	ext{Projector}}}
\mathcal{L}_{LLM}

eq
0
$$

$$

abla_{	heta_{	ext{Evidence}}}
\mathcal{L}_{LLM}

eq
0
$$

$$

abla_{	heta_{	ext{HGNN}}}
\mathcal{L}_{LLM}

eq
0
$$

模型更新建议如下：

| 模块 | 是否更新 |
|---|---:|
| 冻结文本编码器 | 否 |
| Relation-text parameter generator | 是 |
| HGNN 共享 backbone | 是 |
| Label-aware cross-attention | 是 |
| Graph-to-LLM projector | 是 |
| LLM backbone | 否 |
| LLM LoRA / Adapter | 可选 |

这种设置实现了：

> LLM 作为冻结预测器指导图模块学习，而不是由图模块单独训练后再与 LLM 松散拼接。

---

## 8. 总体损失函数

推荐使用以下总体目标：

$$
\mathcal{L}
=
\mathcal{L}_{LLM}
+
\lambda_1\mathcal{L}_{LA}
+
\lambda_2\mathcal{L}_{NT}
+
\lambda_3\mathcal{L}_{STR}
+
\lambda_4\mathcal{L}_{SupCon}
$$

其中 $\mathcal{L}_{SupCon}$ 为可选项。

### 8.1 LLM 分类损失

$$
\mathcal{L}_{LLM}
=
-\log P(y_v\mid v)
$$

作用：使 graph tokens 直接服务于最终 LLM 预测。

### 8.2 标签感知证据对齐损失

$$
\mathcal{L}_{LA}
=
-
\log
rac{
\exp
\left(
\operatorname{sim}(g_{v,y_v},e_{y_v})/	au

ight)
}{
\sum_{c\in\mathcal{Y}}
\exp
\left(
\operatorname{sim}(g_{v,c},e_c)/	au

ight)
}
$$

作用：让正确标签查询到的 evidence 与对应标签语义一致，同时避免直接将基础表示绑定到源域标签空间。

### 8.3 节点—文本对比损失

$$
\mathcal{L}_{NT}
=
\operatorname{CLIPLoss}
\left(
Proj_{	ext{con}}(h_v),
E_{	ext{text}}(\operatorname{text}(v))

ight)
$$

作用：保持基础图表示与节点文本语义一致，增强可迁移性。

### 8.4 结构自监督损失

$$
\mathcal{L}_{STR}
=
-\log
P((u,v,r)\in E\mid h_u,h_v,e_r)
$$

作用：保证模型真实利用关系与拓扑结构。

### 8.5 可选监督对比损失

$$
\mathcal{L}_{SupCon}
$$

作用：提升源域判别能力。

风险：可能产生源域类别偏置，因此建议低权重使用，并通过跨领域消融验证。

---

## 9. 推荐的三阶段训练策略

由于 LLM backward 会产生较高计算与显存消耗，建议采用分阶段训练，而不是从第一步起全程端到端反向传播。

### 阶段一：结构—语义表示预训练

训练模块：

- Relation-Text Parameterized HGNN；
- 节点/关系参数生成器；
- 对比 projection head；
- 结构预测头。

目标函数：

$$
\mathcal{L}^{(1)}
=
\lambda_2\mathcal{L}_{NT}
+
\lambda_3\mathcal{L}_{STR}
$$

目标：

- 让 HGNN 首先学习稳定且通用的结构语义表示；
- 减少训练初期 LLM 信号带来的不稳定性；
- 建立跨 schema 的图编码基础。

### 阶段二：标签感知证据学习

训练模块：

- HGNN；
- Label-Aware Graph Evidence Module；
- 可选 projector 前置层。

目标函数：

$$
\mathcal{L}^{(2)}
=
\mathcal{L}_{LA}
+
\lambda_2\mathcal{L}_{NT}
+
\lambda_3\mathcal{L}_{STR}
$$

目标：

- 训练标签描述如何查询结构证据；
- 使标签相关信息作用在 evidence 层而非污染基础表示层。

### 阶段三：LLM-in-the-loop 对齐微调

训练模块：

- HGNN 的全部或最后若干层；
- Relation-text parameter generator；
- Label-aware Graph Evidence Module；
- Graph-to-LLM Projector；
- 可选 LLM LoRA。

冻结模块：

- LLM backbone；
- 通常冻结文本编码器。

目标函数：

$$
\mathcal{L}^{(3)}
=
\mathcal{L}_{LLM}
+
\lambda_1\mathcal{L}_{LA}
+
\lambda_2\mathcal{L}_{NT}
+
\lambda_3\mathcal{L}_{STR}
$$

目标：

- 用 LLM 最终预测行为直接监督 graph tokens；
- 实现 prediction-driven differentiable alignment；
- 降低全程经过 LLM 反向传播的计算开销。

---

## 10. 推理流程

在目标领域预测节点 $v$ 时：

1. 将目标领域关系描述编码为 relation text embeddings；
2. 由关系文本动态生成消息传递调制参数；
3. 使用 Relation-Text Parameterized HGNN 编码目标节点局部异构子图；
4. 将目标领域候选标签文本编码为 label embeddings；
5. 针对每个标签 $c$，通过 label-aware query 提取 $g_{v,c}$；
6. 投影得到对应 graph tokens $P_{v,c}$；
7. 输入冻结 LLM，计算每个候选标签的 `Yes` 分数；
8. 输出分数最高的标签：

$$
\hat{y}_v
=
rg\max_{c\in\mathcal{Y}}
s_{v,c}
$$

该流程无需为目标领域新建分类头，标签描述本身可作为新标签空间的语义接口。

---

## 11. 实验设计建议

## 11.1 核心研究问题

- **RQ1：** 模型能否在不同异构图 schema 与不同标签空间之间进行有效迁移？
- **RQ2：** Relation-text parameterization 是否优于固定 relation ID 参数或完全同构化 GNN？
- **RQ3：** Label-aware graph evidence 是否优于普通 graph token？
- **RQ4：** LLM prediction loss 反向更新图模块是否优于 post-hoc projector alignment？
- **RQ5：** 节点—文本对比学习与结构自监督是否改善跨领域泛化？
- **RQ6：** 性能提升是否能够抵消 LLM backward 带来的计算代价？

## 11.2 主要对比方法

| 类别 | 方法 | 验证目标 |
|---|---|---|
| LLM-only | LLM + 节点文本 + 标签描述 | 不使用图结构的性能 |
| Textualized Graph LLM | LLM + 序列化邻居/路径/metapath | 对比直接图文本化 |
| GNN/HGNN predictor | GAT / GraphSAGE / HAN / HGT | 对比传统图预测器 |
| Embedding Matching | HGNN embedding + label text similarity | 对比非 LLM predictor |
| Generic Graph Token | HGNN + 普通 graph token + LLM | 验证 label-aware evidence 的必要性 |
| Post-hoc Projector | 独立 HGNN + 后训练 projector + LLM | 验证 prediction-driven alignment |
| Full Model | ReLaHG-LLM | 完整模型 |

## 11.3 关键消融实验

| 消融设置 | 验证内容 |
|---|---|
| w/o relation-text parameterization | 关系文本动态参数生成是否有效 |
| relation ID embedding 替代 relation text | 关系语义是否增强跨 schema 能力 |
| shared homogeneous GNN | 完全同构化是否损失异构关系信息 |
| w/o label-aware graph evidence | 标签查询图证据是否有效 |
| generic graph token 替代 label-aware token | label-aware 机制贡献 |
| w/o LLM backpropagation | 预测驱动对齐是否优于事后对齐 |
| w/o $\mathcal{L}_{NT}$ | 节点—文本对比损失贡献 |
| w/o $\mathcal{L}_{STR}$ | 结构自监督贡献 |
| random/shuffled relation descriptions | 是否真正使用关系语义 |
| shuffled edges | 是否真正使用拓扑结构 |
| node-order perturbation on textualized baseline | 比较序列化图的顺序敏感性 |

---

## 12. 主要贡献点表述

### 贡献一：Prediction-driven Differentiable HGNN–LLM Alignment

本文提出一种区别于 LLM 增强器、图文本序列化推理和事后 projector 对齐的新范式：冻结 LLM 作为最终预测器，通过其分类损失反向优化前端 HGNN 与 graph token projector，使结构表示从训练阶段起就面向 LLM 决策进行学习。

### 贡献二：Relation-Text Parameterized HGNN

本文使用关系自然语言描述动态生成关系条件化消息传递参数，替代传统依赖固定 relation ID 的 schema-specific 参数，使同一套 HGNN backbone 能够处理具有不同关系类型和节点结构的跨领域异构图。

### 贡献三：Label-Aware Graph Evidence Module

本文使用候选标签描述作为语义 query，从 HGNN 结构表示中提取每个候选标签对应的图证据，并将其转化为 graph tokens 输入 LLM，从而实现标签自适应、结构感知的最终预测。

### 贡献四：兼顾泛化与判别的训练目标

本文将节点—文本对比与结构自监督作用于基础结构表示，将标签相关对齐限制在 label-aware evidence 层，以避免基础图空间过度拟合源域标签体系，并提高跨领域迁移能力。

---

## 13. 潜在风险及应对策略

### 13.1 LLM backward 计算量较大

即使冻结 LLM，分类损失仍需经过 Transformer 计算图反传至 graph tokens，导致显存和训练时间增加。

应对方式：

- 采用三阶段训练；
- 仅在最终阶段使用 LLM classification loss；
- 使用 3B/7B 开源 LLM 与 bf16；
- 开启 gradient checkpointing；
- 将 graph token 数控制在 1--4；
- 控制 prompt 长度；
- 若类别数量较多，增加候选标签粗筛选模块。

### 13.2 LLM 可能仅依赖节点文本而忽视图结构

应对方式：

- 比较 LLM-only 与 Full Model；
- 设置 shuffled-edge 和 random graph token 实验；
- 在节点文本不充分的样本子集上评估；
- 加入结构自监督目标与关系文本消融。

### 13.3 源领域标签信息可能损害泛化

应对方式：

- 不直接让基础表示 $h_v$ 与标签文本对齐；
- 仅在 $g_{v,c}$ 上进行 label-aware alignment；
- 限制监督对比损失权重；
- 使用目标领域迁移结果选择超参数。

### 13.4 关系描述质量影响动态参数生成

应对方式：

- 使用规范化的关系描述模板；
- 设计多种等价描述做 ensemble；
- 加入 random/shuffled relation description 对照；
- 设计 unseen-relation transfer 评测。

---

## 14. 可用于论文 Introduction 的核心叙述

现有异构图与大语言模型结合方法通常遵循三种范式：其一，使用 LLM 生成描述、伪标签或提示以增强图模型训练，但最终预测仍由图模型完成，LLM 与 HGNN 之间缺少面向最终决策的紧密对齐；其二，将图结构序列化为文本并直接输入 LLM 进行推理，但这种方式可能损失异构关系与拓扑结构的保真表达，并对 prompt 与输入顺序较为敏感；其三，将独立训练的图 embedding 通过 projector 事后映射到语言空间，但这种几何层面对齐并不保证图表示真正有助于 LLM 的最终预测。

本文认为，异构图与 LLM 融合的关键不在于简单决定由谁预测，而在于让图结构表示成为 LLM 能够直接使用的决策证据。为此，本文提出 ReLaHG-LLM：首先利用关系文本参数化 HGNN，在保留异构消息传递能力的同时避免模型参数绑定于特定 schema；随后，使用候选标签文本作为 query，从图表示中提取标签感知结构证据，并将其投影为连续 graph tokens；最后，将冻结 LLM 置于训练闭环中，以 LLM 分类损失反向优化 HGNN 与 projector。该机制实现了 prediction-driven differentiable alignment，使结构学习、标签语义查询与语言模型决策在统一目标下协同优化。

---

## 15. 一句话总结

> **ReLaHG-LLM 通过“关系文本生成异构消息参数—标签文本查询结构证据—冻结 LLM 完成预测并反向监督图模块”的闭环机制，实现跨 schema、跨标签空间的异构图知识迁移与结构感知语言预测。**
