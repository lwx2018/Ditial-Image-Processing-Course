# Ditial-Image-Processing-Course
2021数字图像处理课程作业

一、lomo滤镜

1.1 运行环境

·VS2017
·opencv-3.4.16-vc14_vc15
·Windows10
1.2 运行步骤

·首先使用vs打开test1.sln
·输入图片的路径，默认为lomo\test1文件中的test2.png
·默认在lomo\test1文件夹中输出结果图像after_lomo.png

1.3算法原理

·使用二维高斯函数进行图像融合，

同时考虑到权重在0-1之间，使用归一化的式子：

使用非线性变换滤波，转换图像。
·把图像风格转变为怀旧风格
转变图像的三通道值，以此达到相应风格效果的目的
·改变图像尺寸至合适大小
可能有些图像过大，不适合视觉上的处理，由此，设立一个阈值，超出此阈值的图像调整其尺寸。

1.4算法效果

<img width="415" alt="image" src="https://user-images.githubusercontent.com/40064484/142761848-88d87555-911d-4182-acc0-aa39ef425578.png">

滤波前
<img width="416" alt="image" src="https://user-images.githubusercontent.com/40064484/142761856-63f81499-0659-45b1-b1f0-d2f257b6b3f7.png">

滤波后

二、人像美肤

2.1 运行环境

·VS2017
·opencv-3.4.16-vc14_vc15
·Windows10

2.2 运行步骤

·首先使用vs打开test.sln
·输入图片的路径，默认为skin\test文件中的test.png
·默认在skin\test文件夹中输出结果图像skin.png

2.3算法原理

·使用二维高斯函数进行图像融合，

同时考虑到权重在0-1之间，使用归一化的式子：

·双边滤波
计算空间距离，（xi，yi）为当前点位置，（xc，yc）为中心点位置，sigma为空间域标准差。

计算灰度距离，gray（xi，yi）为当前点灰度值，gray（xc，yc）为中心点灰度值，sigma为灰度值标准差。


2.4算法效果

滤波前

滤波后

参考资料
https://xta0.me/2013/11/20/iOS-Lomo-Effect.html
https://www.cnblogs.com/yuxi-blog/p/9959304.html
http://wiki.opencv.org.cn/index.php/Cxcore%e7%bb%98%e5%9b%be%e5%87%bd%e6%95%b0#Ellipse
