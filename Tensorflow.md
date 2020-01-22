TensorFlow的使用

	1. TensorFlow的结构
 	2. TensorFlow的各个组件
      	1. 图
      	2. 会话
      	3. 张量
      	4. 变量
	3. 简单的线性回归案例 



TensorFlow结构分析

1. 流程图：定义数据（张量Tensor）和操作（节点Operations）
2. 执行图：调用各方资源，将定义好的数据和操作进行起来



查看默认图的方法：

	1. 调用方法： 用tf.get_default_graph()
 	2. 查看属性： .graph

创建图：

	1. new_g = tf.Graph()
 	2. with new_g.as_default(): 定义数据和操作

TensorBoard可视化图结构：

 1. 数据序列化-events文件：

     1. ```
        tf.summary.FileWriter("./temp/summary", graph=sess.graph)
        ```

     2. tensorboard --logdir="./summary/" --host=127.0.0.1

     3. http://127.0.0.1:6006

	2. 变量显示：

    	1. 收集变量：

        	1. tf.summary.scalar(name='', tensor)收集对于损失函数和准确率等单值变量， 那么为变量的名字，tensor为值
        	2. tf.summary.histogram(name='', tensor) 收集高维度的变量参数
        	3. tf.summary.image(name = '', tensor) 收集输入的图片张量能显示图片

    	2. 合并变量写入事件文件

        	1. merged = tf.summary.merge_all()

        	2. 运行合并： summary = sess.run(merged)

        	3. 每次迭代加入

            

会话的run():

1. run(fetches, feed_dic = None, options = None, run_metadata = None): fetches can be list, tuple
2. sess = tf.Session(); print(less.run(c)); print(c.eval(session = sess))
3. feed: placeholder提供占位符，run时候通过feed_dict指定参数



TensorFlow Api:

1.  基础API： tf.app; tf.image; tf.fgile; tf.python_io; tf.train; tf.nn;
2. 高级API：tf.keras; tf.layers; tf.contrib; tf.estimator



TensorFlow 线性回归：

1. Review:
   1. model: linear regression
   2. loss function: MSE
   3. optimize loss function: GD
2. 流程分析：
   1. 准备数据： 100 sample; features: [100,1]; target : y_predict = [100,1]; X* w([1,1]) + bias(1,1) 
   2. 构造模型：y_predict = tf.matmul(x, weight)
   3. 构造loss function: error = tf.reduce_mean(tf.square(y_predict- y_true))
   4. 优化损失：optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(error)



案列代码注意点：

1. learning_rate： 
   1. weights go to Nan, what we can do :
      1. redesigh the network
      2. change learning rate
      3. 使用激活函数
      4. 使用梯度函数

模型的保存与加载：

1. saver = tf.train.Saver(var_list = None, max_to_keep = 5)
2. saver.save(sess, path) : saver.save(sess, './temp/model/model.ckpt')
3. saver.restore(sess, path)



命令行参数使用：

1. tf.app.flags.DEFINE_integer("max_step", 0, "explaination")
2. tf.app.flags.DEFINE_string("model_dir", " ", "explaination")
3. FLAGS = tf.app.flags.FLAGS
4. python tensor.py —max_step = 200 —model_dir = "hello world"



数据IO操作：

1. 占位符 & feed_dict 搭配使用

2. queueRunner：

   1. 通用文件读取流程：多线程 + 队列

      1. 构造文件名队列：file_queue = tf.train.string_input_produce(string_tensor, shuffle = True)

      2. 读取与解码

         

         1. 文本：
            1. 读取： tf.TextLineReader()
            2. 解码 :tf.decode_csv:
         2. 图片：
            1. 读取：tf.WholeFileReader()
            2. JPEG图片解码：tf.image.decode_jpeg(contents)
            3. PNG图片解码：tf.image.decode_png(contents)
         3. 二进制：
            1. 读取： tf.FixedLengthRecordReader()
            2. 解码二进制文件内容：tf.decode_raw
         4. TFRecords: tf.TFRecordReader(): key, value = 读取器.read(file_queue); key:文件名； value： 一个样本

      3. 批处理队列：

         1. tf.trai.batch(tensors, batch_size, num_threads = 1, capacity=32, name = None)
         2. tf.train.shuffle.batch

      4. 手动开启线程： tf.train.QueueRunner

         1. 开启会话：tf.train.start_queue_runners(sess= None, coord = None);
         2. tf.train.Coordinator():线程协调员

   2. 图片数据：

      1. 图像基本知识：
         1. 特征提取：
            1. 文本-数值（二维数值shape(n_samples, m_features)）
            2. 字典-数值（二维数值shape(n_samples, m_features)）
            3. 图片-数值（三维数值shape(图片长度，图片宽度，图片通道数)）组成图片的基本单位是像素
         2. 图片三要素：图片长度，图片宽度，图片通道数
            1. 灰度图[长，宽，1]：每一个像素点[0, 255], 0 黑，255 白
            2. 彩色图[长，宽，3]：每一个像素点用3个[0, 255]的数
         3. 张量形状：Tensor(指令名称，shape，dtype)
            1. 单个图片：[height, width, channel]
            2. 多个图片：[batch， height, width, channel]
         4. 图片特征值处理：
            1. 图片缩放到统一大小：tf.image.resize_image(images, size)
         5. 数据格式：
            1. save: uint8
            2. 矩阵计算：float32
      2. 图片案例分析：
         1. 构造文件名队列
         2. 读取与解码：使样本的形状和类型统一
         3. 批处理
      3. 二进制数据读取：
         1. reader = tf.FixedLengthRecordReader(3073)
         2. key,value  = reader.read(file_queue)
         3. decoded = tf.decode_raw(value, tf.unit8)
         4. 对tensor对象进行切片： label 一个样本image(3072字节 = 1024r+1024g+1024b)
         5. [[r[32,32]],[g[32,32]],[b[32,32]]], shape = (3, 32, 32) = (channels, height, width) ==>TensorFlow的图像表示习惯
         6. 批处理



卷积神经网络：

1. 卷积层(convolutional layer)

   1. 卷积核-filter - 过滤器 - 卷积单元 - 模型参数

      1. 个数
      2. 大小 $1*1$， $3*3$$, $ 5*5： 卷积如何计算： 输入 5*5*1 filter 3*3*1
      3. 步长
      4. 零填充的大小

   2. 输出结果的大小：

      1. 输入体积大小$H_1 \times W_1 \times D_1$
      2. 四个超参数：
         1. Filter 数量K
         2. filter大小F
         3. 步长S
         4. 零填充大小P
      3. 输出体积大小$H_2 \times W_2\times D_2$
         1. $H_2 = (H_1- F + 2P)/S+1$
         2. $W_2 = (W_1- F + 2P)/S+1$
         3. $D_2 =K$
         4. 例子：输入图像为 32 * 32 * 1 ， 50个filter， 大小为5 * 5， 移动步长为1， 零填充大小为 1，请输出大小？

   3. 多通道图片如何观察：

      1. 输入图片： 7 * 7 * 3
      2. filter 数量 ： 2
      3. filter大小： 3 * 3 * 3
      4. 步长： 2

   4. 卷积网络API:

      1. tf.nn.conv2d(input, filter, strides, padding)

      2. input: [batch, heigth, width, channel], float32, 64

      3. filter:  weights + bias; initial_value = random_normal(shape = [filter_height, filter_width, in_channels（彩色3， 黑白1）, out_channels(filter 的数量) ]);

         [F, F, 3 or 1,  K]

      4. strides: 步长；strides = [1, stride, stride, 1]

      5. padding:填充； ”same“: 越过边沿取样： ceil（H/S）,"VALID"：不越过边沿取样（losing data）

2. 激活函数：

   1. sigmoid: $\frac{1}{1+e^{-x}}$, if the |value| >6, the result is the same, 计算量大 ； 在反向传播的时候，梯度消失
   2. ReLU = max(0, x): 解决梯度消失问题，计算快，SGD（批梯度下降）， 图像没有负值: tf.nn.relu(features, name= None)

3. polling 池化层：

   1. 特征提取： 利用图像上像素点之间的联系，去掉feature map中不重要的样本，进一步减少参数数量， max_polling; avg_polling
   2. tf.nn.max_pool(value, size, strides, padding)
      1. value:[batch, height, width, channels]
      2. ksize = 池化层窗口大小[1,ksize, ksize, 1]
      3. strides : [1,strides, strides, 1]
      4. padding: default is "same"
   3. full connection 全连接层作为分类器的作用
   4. example：输入图片为200 * 200， 依次经过一层卷积（kernel size 5 * 5, padding 1, stides 2), pooling (kernel size 3 * 3, padding 0, stride 1), 又一层卷积 （kernel size 3 * 3， padding 1， stride 1）之后，输出特征图大小为：A95， B96， C97， D98， E99， F100
      1. 提示： 卷积向下取整， 池化向上取整
      2. $H_2 = (H_1- F + 2P)/S+1$
      3. $W_2 = (W_1- F + 2P)/S+1$
      4. $D_2 =K$
      5. (200-5+2)/2+1 = 99, 99, 1, —>(99 - 3)/1 +1=97, 97,1 —>(97-3+2)/1+1 = 97, 97, 1
      6. answer: C

4. 案例:

   1. 设计网络：卷积层，激活层，池化层，卷积层，激活层，池化层， 全连接层
   2. 具体参数：
      1. 第一层
         1. 卷积：32 个filter， 大小5  * 5, strides= 1, padding = "same"
            1. input:输入图像[None, 28, 28 ,1]
            2. tf.nn.conv2d(input, filter, strides, padding)
            3. filter:
               1. weights = tf.Variable(initial_value = random_normal(shape = [5, 5, 1, 32]))
               2. bias = tf.Variable(initial_value = random_normal(shape = [32]))
            4. strides: 1: [1,1,1,1]
            5. padding: "Same"
            6. 输出形状：[None, 28, 28, 32]
         2. 激活： ReLU
         3. 池化：大小2 * 2， strides= 2
      2. 第二层
         1. 卷积：64个filter， 大小5  * 5, strides= 1, padding = "same"
            1. input:输入图像[None, 14，14，32]
            2. tf.nn.conv2d(input, filter, strides, padding)
            3. filter:
               1. weights = tf.Variable(initial_value = random_normal(shape = [5, 5, 32, 64]))
               2. bias = tf.Variable(initial_value = random_normal(shape = [42]))
            4. strides: 1: [1,1,1,1]
            5. padding: "Same"
            6. 输出形状：[None, 14, 14, 64]
         2. 激活： ReLU
         3. 池化：大小2 * 2， strides= 2:
            1. 输入形状：[None, 14, 14, 64]
            2. 输出形状：[None, 7, 7， 64]
      3. 全连接：
         1. tf.reshape()—>[None, 7, 7, 64]
         2. y_predict = tf.matmul(pool2, weights  )+bias
      4. optimize and improve --调参：
         1. learning rate
         2. 随机初始化的值
         3. other optimizer
         4. 对于深度网络使用batch normalization(在使用的这一层输出的模型参数分布保持一致) 或者droupout（使得某些神经元失效，模型参数消失，降低模型复杂度）— fix overfit







案例：验证码识别的案例

1. 数据集

2. 对数据集中的特征值和目标值怎么样

3. 如何分类？

   1. 如何比较真实值和输出值的正确性？
   2. 如何衡量损失？：
      1. 手写数字识别用的softmax + cross entropy
      2. 一个样本对应4个目标值：sigmoid + cross entropy
   3. 准确率的计算：
      1. 核心： 对比真实值和预测值最大值所在的位置
      2. 手写数字识别： y_predict[None,10], tf.argmax(y_predict, axis = 1)
      3. y_predict[None,4,26], tf.argmax(y_predict, axis = 2/-1)
      4. tf.reduce_all() 

4. 流程分析

   1. 读取图片数据 filename—》标签值

   2. 解析csv文件， 讲标签值NZPP-[13,25,15,15]

   3. 讲filename和标签值联系起来

   4. 构建卷积神经网络 -》y_predict

   5. 构建损失函数

   6. 优化损失

   7. 计算准确率

   8. 开启会话，开启线程

      

5. 代码实现





y

















