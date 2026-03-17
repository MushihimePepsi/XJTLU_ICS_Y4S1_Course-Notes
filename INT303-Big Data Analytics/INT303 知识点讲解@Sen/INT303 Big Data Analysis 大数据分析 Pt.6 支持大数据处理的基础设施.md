@[toc]
# 1. 大规模计算（Large-scale computing）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e058e1ec36034eabb1a14c2e7583e185.png)

## 1.1 单节点架构（Single Node Architecture）
我们平时用的都是单节点架构。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f4e32848fe6b4ba799838608ebc9c4f5.png)
CPU负责执行计算任务，Memory（内存）用于临时存储数据和程序，以便CPU快速访问，磁盘用于长期存储数据。
单节点架构可以用于执行机器学习和统计分析任务。这些任务通常需要在内存中处理大量数据，因此内存的大小和CPU的处理能力是关键因素。
传统的数据挖掘任务也可以在单节点架构上执行。这些任务通常涉及从磁盘读取数据，然后在内存中进行处理和分析。

## 1.1 集群架构（Cluster Architecture）
这是一种用于处理大规模数据和高性能计算的分布式系统架构。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/72a11a40febe481e9380055b4465f043.png)
Switch（交换机）：
交换机用于连接不同的计算节点，并管理它们之间的数据传输。
Rack（机架）：
每个机架包含多个计算节点（图中显示为16-64个节点）。
Node（节点）：
每个节点包含基本的计算组件：CPU、内存（Mem）和磁盘（Disk），也就是前面的单节点架构。

机架内连接：
机架内任意两个节点之间的连接速度为1 Gbps（千兆比特每秒）。
这种高速连接确保了机架内节点之间的快速数据传输。
机架间连接：
机架之间的连接速度为2-10 Gbps（2到10千兆比特每秒）。
这种背板连接（backbone）提供了集群中不同机架之间的高速数据传输。

集群架构通过将多个计算节点连接在一起，形成一个强大的分布式计算系统。它利用高速网络连接（如1 Gbps和2-10 Gbps）确保节点之间的快速数据传输，从而实现高效的并行计算。这种架构特别适合处理大规模数据和需要高性能计算的应用场景。
2011年估计Google拥有约100万台机器。

集群架构在物理上是由多个单节点架构组成，但它在逻辑上是一个整体，通过软件和网络技术实现节点间的紧密协作。
# 2. 分布式文件系统（Distributed file system）
在大规模计算环境中，数据量和计算需求通常非常庞大，单台机器无法处理。因此，需要将任务分解成小块，然后分配给多个节点并行处理，以提高计算效率和速度。所以我们要思考如何分布计算。
分布式程序设计通常比单机程序设计更复杂，因为它涉及到多个节点之间的通信、数据同步、任务协调等问题，所以如何简化我们需要考虑。
在大规模计算环境中，硬件故障是不可避免的。因此我们要考虑容错机制和故障恢复策略。

在网络中传输数据需要时间，这在大规模数据处理中是一个显著的性能瓶颈。当数据量很大时，通过网络移动数据可能会导致显著的延迟，从而影响整体计算效率。
与其将大量数据传输到计算资源，不如将计算任务移动到数据所在的位置。这样可以减少数据传输的需求，从而节省时间和带宽。
为了提高数据的可靠性和容错能力，可以在多个位置存储数据的副本。这样即使某个存储位置发生故障，数据仍然可以从其他副本中恢复。
因此我们推出MapReduce，这是一种编程模型，用于大规模数据集的并行处理。它通过Map（映射）和Reduce（归约）两个步骤来处理数据，使得开发者可以更容易地编写分布式程序。
Apache Spark和Hadoop是两个流行的开源框架，它们提供了工具和库来支持大规模数据处理和分布式计算。这些框架利用上述想法来优化数据处理流程。

而对于这些我们需要分布式文件系统。
流行的有两种：
Google: GFS：Google File System（GFS）是Google开发的一个用于大规模数据存储的分布式文件系统。它设计用于高可靠性和高吞吐量的数据访问。
Hadoop: HDFS：Hadoop Distributed File System（HDFS）是Hadoop项目的一部分，它是一个高度容错的系统，设计用于在廉价硬件集群上存储和处理大量数据。

分布式文件系统提供了一种在多个节点上存储和访问数据的方法，即使某些节点发生故障，数据仍然可以被访问和恢复。
它提供全局文件命名空间（Provides global file namespace），用户可以像访问本地文件系统一样访问分布式文件系统中的文件，而不需要关心文件实际存储在哪个节点上。全局文件命名空间为用户和应用程序提供了一个统一的视图，简化了数据访问和管理。

## 2.1 基本架构
分布式文件系统包含三个部分：
1. 块服务器（Chunk Servers）
在分布式文件系统中，文件被分割成多个连续的数据块（chunks），每个块通常大小为16-64MB。
为了提高数据的可靠性和容错能力，每个数据块会被复制多次（通常是2倍或3倍）。
为了进一步提高数据的安全性，副本通常会被存储在不同的物理位置，如不同的机架，以防止单点故障。
2. Master Node（主节点）
在Hadoop的HDFS中称为NameNode（a.k.a. Name Node in Hadoop’s HDFS）：
主节点负责管理文件系统的元数据，包括文件的块信息和块的位置。
主节点记录了文件系统中所有文件的元数据，包括文件被分割成哪些块以及这些块存储在哪些节点上。
为了提高系统的可靠性，主节点的元数据可能会被复制到其他节点上。
3. 文件访问客户端库（Client Library for File Access）
客户端库负责与主节点通信，以获取文件块的位置信息。
一旦客户端知道了文件块的位置，它可以直接连接到存储这些块的块服务器来读取或写入数据。

下图展示了分布式文件系统。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/acadd82e68ff4a0186a9a9732f0dece1.png)
数据保存在“块”中并分布在多台机器上（Data kept in “chunks” spread across machines）。
每个块在不同的机器上复制（Each chunk replicated on different machines）。
从磁盘或机器故障中无缝恢复（Seamless recovery from disk or machine failure）。

块服务器不仅负责存储数据，还可以执行计算任务。这种设计使得计算和存储紧密结合，进一步提高了系统的性能和效率。
因此计算任务可以直接在存储数据的块服务器上执行，而不是将数据移动到计算节点上。这样可以减少数据传输的开销，提高计算效率。

# 3. MapReduce:分布式计算编程模型（MapReduce:Distributed computing programming model)
MapReduce是一种编程范式，主要用于处理和生成大数据集的并行计算。
MapReduce通过将计算任务分解为两个主要步骤（Map和Reduce）来简化并行编程。Map阶段将输入数据分割成小块并进行处理，而Reduce阶段则将Map阶段的输出结果进行汇总和合并。这种分阶段的处理方式使得并行计算更加容易实现。
MapReduce框架自动处理节点故障，包括硬件故障和软件故障。如果某个节点失败，框架会自动重新分配任务到其他节点，从而确保计算任务的完成。这种容错机制使得开发者无需手动管理故障。
MapReduce框架能够有效地管理和处理大规模数据集。它通过将数据分割成小块并在多个节点上并行处理这些数据块，从而简化了大规模数据的管理。

MapReduce有多种实现，包括：Hadoop、Spark、Flink、Google的原始版本。我们后面将介绍Spark。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6fefa78ecce74b749ecfac8dcdcaf15f.png)
输入（Input）：输入数据被分割成多个块，每个块由Map任务处理。
Map阶段（MAP）：Map函数读取输入数据，并将其转换为一组键值对（key-value pairs）。每个键值对包含一个键（key）和一个值（value）。
中间结果（Intermediate）：在Map阶段产生的键值对被输出，这些输出是中间结果，等待被进一步处理。
按键分组（Group by key）：这些中间结果被“洗牌”（shuffle），即重新组织和分配，以便所有具有相同键的键值对被收集到一起。这个过程包括哈希合并（Hash merge）、排序（Sort）和分区（Partition）。
归约阶段（Reduce）：Reduce函数接收分组后的键值对，对每个键的所有值进行处理（例如，求和、平均、合并等），并生成最终的输出结果。
输出（Output）：最终的输出结果是经过Reduce阶段处理后的数据，这些数据可以是统计信息、汇总数据或其他形式的分析结果。

## 3.1 主要步骤
我们再仔细讲述一下Map Reduce处理数据的三个主要步骤：
1. Map（映射）：
在Map阶段，用户定义的Map函数被应用到输入数据的每个元素上。这一步是Mapper应用Map函数到单个元素。多个Mapper被组织成一个Map任务，这是并行处理的基本单位。Map函数的输出是一组0个、1个或多个键值对。
2. Group by Key（按键分组）：
系统将所有的键值对按照键进行排序，并将具有相同键的键值对分组在一起，输出为键-值列表（key-(list of values)）对。
3. Reduce（归约）：
在Reduce阶段，用户定义的Reduce函数被应用到每个键及其对应的值列表上。Reduce函数接收一个键和该键的所有值，然后对这些值进行处理，生成最终的输出结果。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4400020c41f4c909ba88e6801476051.png)

下图清晰地展示了MapReduce的Map步骤如何工作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6eba65977b104e21aa0bb7b631eb82a9.png)
输入数据被分割成多个部分，每个部分由一个Map任务处理。
Map任务将输入数据转换为中间的键值对。
这些键值对随后会被传递到Reduce阶段，进行进一步的处理和汇总。

下图清晰地展示了MapReduce的Reduce步骤如何工作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f08165c42f314d739ef3d31b317bb849.png)
按键分组：将所有具有相同键的键值对分组在一起。
应用Reduce函数：对每个键的所有值进行处理，生成最终的输出结果。
输出结果：Reduce函数的输出是MapReduce作业的最终结果。

下图展示了一个执行架构。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d3f6e511a00449398ad8bb620631a8c.png)
1. 用户程序（User Program）：用户程序是启动MapReduce作业的入口点。它通过调用MapReduce API来定义Map和Reduce函数，并指定输入和输出。
2. 主节点（Master）：MapReduce作业的协调者。它负责管理整个作业的执行流程，包括：
接收用户程序提交的作业。
将作业分解为多个Map任务和Reduce任务。
为每个任务分配工作节点。
监控任务的执行状态，并处理失败的任务。
3. 工作节点（Worker）：实际执行Map和Reduce任务的节点。它们从主节点接收任务分配，并执行以下操作：
对于Map任务，工作节点读取输入数据，应用Map函数，生成中间键值对。
对于Reduce任务，工作节点从Map任务收集中间键值对，按键分组，应用Reduce函数，生成最终输出。
4. 输入数据（Input Data）：输入数据是MapReduce作业的原始数据源。它被分割成多个块，并由多个工作节点并行读取和处理。
5. 中间文件（Intermediate Files）：中间文件是Map任务的输出结果。每个Map任务生成一个中间文件，这些文件随后被Reduce任务读取和处理。
6. 输出文件（Output File）：输出文件是Reduce任务的最终结果。Reduce任务将处理后的数据写入输出文件，完成整个MapReduce作业。

## 3.2 概念练习
我们做一些判断题来回顾一下刚刚的知识。
1. 每个mapper/reducer必须生成与它在输入中接收到的相同数量的输出键/值对。
这个说法是错误的。在MapReduce中，mapper和reducer产生的输出键值对的数量取决于具体的处理逻辑，而不是必须与输入数量相同。例如，mapper可能会根据某些条件过滤掉一些数据，或者将一个输入生成多个输出。
2. keys/values of mappers/reducers输出类型必须与其输入类型相同。
这个说法也是错误的。mapper和reducer的输出键值对的类型不必与输入类型相同。实际上，mapper和reducer通常用于转换数据格式或提取信息，因此输出类型可能会与输入类型不同。
3. reducers的输入是按键分组的。
这个说法是正确的。在MapReduce中，所有具有相同键的键值对会被分组在一起，然后传递给同一个reducer进行处理。这是MapReduce模型的一个关键特性，它允许对相同键的所有值进行聚合或汇总操作。
4. 在一些mappers仍在运行时，可以启动reducers。
这个说法是错误的。在标准的MapReduce模型中，所有的mapper必须完成它们的任务，并且所有的中间结果（shuffle和sort阶段）必须准备好之后，reducer才能开始工作。这是因为reducer需要所有与特定键相关的数据才能进行处理。

## 3.3 示例
我们现在通过一个例子来展示MapReduce模型。
我们现在想要以单词计数。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6924cd4b79854662a1c4b8188dc3fb58.png)
我们需要思考如何编写Map函数读取输入文档的每个单词，还要考虑编写Reduce函数接收每个键（单词）及其对应的值列表（出现次数）并输出该键（单词）及其总出现次数。
对于Map函数的代码如下：

```python
map(key, value):
    # key: document name; value: text of the document
    for each word w in value:
        emit(w, 1)
```
Map函数读取输入文档的文本内容，并对每个单词进行处理。
对于文本中的每个单词w，Map函数生成一个键值对(w, 1)。
emit(w, 1)表示输出一个键值对，其中键是单词w，值是1，表示该单词出现一次。

对于Reduce函数的代码如下：

```python
reduce(key, values):
    # key: a word; value: an iterator over counts
    result = 0
    for each count v in values:
        result += v
    emit(key, result)
```
Reduce函数接收一个单词及其对应的出现次数列表。
初始化一个变量result为0，用于累加该单词的总出现次数。
遍历values中的每个计数v，并将它们加到result中。
emit(key, result)表示输出最终的键值对，其中键是单词，值是该单词在所有文档中的总出现次数。

我们再看一个例子，使用MapReduce模型统计列表中每个食品项被多少人选择的情况。
输入文件是一个表格如下图所示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97ab596a33144dd58e75f6f8748f17c6.png)
Map阶段（Map process turns）
输入（Input）：
每一行输入数据来自输入列表文件，这个文件包含了每个人的食品项选择。
输出（Output）：
输出是食品项作为键（key），以及值为1的计数。这意味着Map函数为列表中的每个食品项生成一个键值对，其中键是食品项，值是1，表示该食品项被选择了一次。

Reduce阶段（Reduce process turns）
输入（Input）：
每个输入是来自Mappers的键值对，其中键是食品项，值是一个列表。这个列表包含了该食品项在Map阶段被计数的所有1。
输出（Output）：
输出是键值对，其中键是食品项，值是值的总和（即人数）。Reduce函数接收每个食品项的所有计数，将它们相加，得到该食品项被选择的总次数。

这两个例子其实我们都可以用字典这种数据结构来形象地理解，但是实验MapReduce可以实现并行执行，以处理大规模数据集。
# 4.Spark：扩展了MapReduce（Spark: Extends MapReduce）
Apache Spark是一个开源项目，由Apache软件基金会管理。 它是一个开源的分布式计算系统，它扩展了MapReduce模型并提供了更高级的功能和灵活性。
Spark支持多种编程语言，包括Java、Scala和Python。这使得开发者可以根据自己的偏好或项目需求选择合适的语言进行开发。
它设计一个关键概念：弹性分布式数据集（RDD），这是Spark中的核心数据结构，它是一个分布式的、不可变的、容错的数据集合。RDD允许开发者在集群中并行操作数据，同时提供了数据容错机制，即在数据丢失时可以从其他副本中恢复。
除此之外Spark提供了多种高级API，这些API用于处理不同类型的数据聚合操作，如SQL查询、流处理、机器学习等由于Spark提供了多种数据处理API，它也支持SQL查询。

## 4.1 Hadoop MapReduce和Apache Spark的RDD的比较
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2150189422a44c0297cbc73de922cf73.png)

Hadoop MapReduce是一个基于磁盘的计算框架，它使用Hadoop分布式文件系统（HDFS）来存储数据。
在MapReduce模型中，数据首先通过Map任务进行处理，然后通过Reduce任务进行汇总。
每个步骤的中间结果都写入到磁盘上，这可能会成为性能瓶颈，因为磁盘I/O相对较慢。
流程：
1. 用户提交MapReduce作业。
2. 数据从HDFS读取并分配给多个Map任务。
3. Map任务处理数据并生成中间结果，这些结果被写入到磁盘。
4. Reduce任务读取中间结果，进行汇总处理，并生成最终输出。
5. 最终输出被写入到HDFS。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d9b2a9233744144b65a750f8dda29a8.png)

Apache Spark使用RDD作为其核心数据结构，这是一种分布式内存计算模型。
RDD允许数据在内存中进行计算，这大大提高了数据处理速度。
流程：
1. 创建RDD：从数据源（如HDFS、S3等）加载数据并创建RDD。
2. 转换（Transformation）：对RDD进行转换操作，如map、filter、join等。这些操作是懒惰的（lazy），即它们不会立即执行，而是构建一个新的RDD。
3. 行动（Action）：执行行动操作，如count、collect等，这些操作会触发实际的计算并生成结果。
4. 结果：计算结果可以被收集到驱动程序中或写入到外部存储系统。