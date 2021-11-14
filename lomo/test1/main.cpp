#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>
#include <math.h>


using namespace cv;
using namespace std;


Mat OldschoolStyle(Mat inputimage,Mat outputimage) //转换图像风格转为怀旧风格
{
	
	for (int i = 0; i < inputimage.rows; i++)//.rows是读入的图片的行像素值
	{
		for (int j = 0; j < inputimage.cols; j++) //.cols是读入的图片的列像素值
		{
			outputimage.at<Vec3b>(i, j)[0] = 10* sqrt(inputimage.at<Vec3b>(i, j)[0]);//新图像蓝色通道值重构
			outputimage.at<Vec3b>(i, j)[1] = (inputimage.at<Vec3b>(i, j)[1]);//新图像绿色通道值重构
			outputimage.at<Vec3b>(i, j)[2] = (inputimage.at<Vec3b>(i, j)[2]);//新图像红色通道值重构
		}

	}
	return outputimage;
}


Mat Gussiblu(Mat inputimage1, Mat inputimage2, Mat outputimage)//滤波器实现
{
	int rx = 0;
	int ry = 0;
	int sigma = 600;
	/*double **wbyte = 0;*/
	vector<vector<double> >wbyte(inputimage1.rows, vector<double>(inputimage1.cols));
	int c = sigma ^ 2;
	for (int i = 0; i < inputimage1.rows; i++)
	{
		for (int j = 0; j < inputimage1.cols; j++)
		{
			rx = i - inputimage1.rows / 2;
			ry = j - inputimage1.cols / 2;
			int a = rx ^ 2;
			int b = ry ^ 2;
			
			wbyte[i][j] = exp(((-0.5) * (a+b)) / c);
		}
	}
	/*int wx = 0;
	int wy = 0;*/
	float w = 0.0f;
	/*for (int k = 0; k < inputimage1.rows * inputimage1.cols; k += 4)
	{
		wx = (k % inputimage1.rows) / 4;
		wy = k / inputimage1.rows;*/
		
		for (int i = 0; i < inputimage1.rows; i++)
		{
			for (int j = 0; j < inputimage1.cols; j++)
			{
				w = wbyte[i][j];
				outputimage.at<Vec3b>(i, j)[0] = inputimage1.at<Vec3b>(i, j)[0] * w + inputimage2.at<Vec3b>(i, j)[0] * (1 - w);//新图像蓝色通道值重构
				outputimage.at<Vec3b>(i, j)[1] = inputimage1.at<Vec3b>(i, j)[1] * w + inputimage2.at<Vec3b>(i, j)[1] * (1 - w);//新图像绿色通道值重构
				outputimage.at<Vec3b>(i, j)[2] = inputimage1.at<Vec3b>(i, j)[2] * w + inputimage2.at<Vec3b>(i, j)[2] * (1 - w);//新图像红色通道值重构
			}
		}
		

	/*}*/


	return outputimage;
}

Mat RoiReverse(Mat inputimage,Mat outputimage) {
	
	Mat roi, res,reverse_res, reverse_roi;
	roi = Mat::zeros(inputimage.size(), inputimage.type());
	res = Mat::zeros(inputimage.size(), inputimage.type());
	reverse_res = Mat::zeros(inputimage.size(), inputimage.type());
	reverse_roi = Mat::zeros(inputimage.size(), inputimage.type());//创造一张与原图像大小和类型相同的空白图像，像素值初始化为0，即为黑色图像
	for (int i = 0; i < reverse_roi.rows; i++)//.rows是读入的图片的行像素值；整个图像变为白色
	{
		for (int j = 0; j < reverse_roi.cols; j++) //.cols是读入的图片的列像素值
		{
			reverse_roi.at<Vec3b>(i, j)[0] = 255;//新图像蓝色通道值重构
			reverse_roi.at<Vec3b>(i, j)[1] = 255;//新图像绿色通道值重构
			reverse_roi.at<Vec3b>(i, j)[2] = 255;//新图像红色通道值重构
		}

	}
	/*imshow("roi",roi);
	imshow("reverse_roi", reverse_roi);*/
	//画椭圆roi区域
	int thickness = 2;
	int linetype = 8;
	double angle = 90;
	double w = inputimage.cols;
	double v = inputimage.rows;
	ellipse(roi, Point(w / 2.0, 2 * v / 3.0), Size(2 * v /3.0, v / 3.0), angle, 0, 360, Scalar(255, 255, 255), thickness, linetype);//黑色图像画白线，即黑底白线椭圆
	/*imshow("roi——ellipse", roi);*/
	ellipse(reverse_roi, Point(w / 2.0, 2 * v / 3.0), Size(2 * v / 3.0, v / 3.0), angle, 0, 360, Scalar(0.0,0), thickness, linetype);//白色图像画黑线，即白底黑线椭圆
	/*imshow("reverse_roi", reverse_roi);*/
	floodFill(reverse_roi, Point(w / 2.0, v / 2.0), Scalar(0, 0, 0));//填充一下椭圆区域内的颜色为白色，此时图像为白底黑色实心椭圆
	/*imshow("reverse_roi", reverse_roi);*/
	floodFill(roi, Point(w / 2.0, v / 2.0), Scalar(255, 255, 255));//填充一下椭圆区域内的颜色为黑色，即为黑底白实心椭圆
	/*imshow("roi", roi);*/
	
	inputimage.copyTo(res, roi);
	/*imshow("取部分关注的地方",res);*/
	inputimage.copyTo(reverse_res, reverse_roi);
	/*imshow("取部分不关注的地方", reverse_res);*/
	/*cvtColor(res, res, CV_BGR2HSV);*/
	/*imshow("转换色调", res);*/
	for (int row = 0; row < res.rows; row++)//改变res图像的亮度和对比度使用两个参数来改变分别是阿尔法和贝塔，有时分别控制对比度和亮度，阿尔法必大于零
	{
		for (int col = 0; col < res.cols; col++)
		{
			double alpha = 1.2;
			double beta = 12;
			float b = res.at<Vec3b>(row, col)[0];
			float g = res.at<Vec3b>(row, col)[1];
			float r = res.at<Vec3b>(row, col)[2];
			res.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(alpha * b + beta);
			res.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(alpha * g + beta);
			res.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(alpha * r + beta);
		}
	}
	

	/*imshow("转换结果1", res);*/
	/*cvtColor(reverse_res, reverse_res, CV_BGR2HSV);*/
	/*imshow("转换色调", reverse_res);*/
	for (int row = 0; row < reverse_res.rows; row++)//改变reverse_res图像的亮度和对比度使用两个参数来改变分别是阿尔法和贝塔，有时分别控制对比度和亮度，阿尔法必大于零
	{
		for (int col = 0; col < reverse_res.cols; col++)
		{
			double alpha = 1.25;
			double beta = -10;

			float b = reverse_res.at<Vec3b>(row, col)[0];
			float g = reverse_res.at<Vec3b>(row, col)[1];
			float r = reverse_res.at<Vec3b>(row, col)[2];
			reverse_res.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(alpha * b + beta);
			reverse_res.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(alpha * g + beta);
			reverse_res.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(alpha * r + beta);
			/*reverse_res.at<Vec3b>(y, x)[1] = 1.5*reverse_res.at<Vec3b>(y, x)[1];
			if (reverse_res.at<Vec3b>(y, x)[1] > 255)
			{
				reverse_res.at<Vec3b>(y, x)[1] = 255;
			}
			reverse_res.at<Vec3b>(y, x)[2] = reverse_res.at<Vec3b>(y, x)[2];*/
		}
	}

	/*cvtColor(reverse_res, reverse_res, CV_HSV2BGR);*/
	/*imshow("转换结果2", reverse_res);*/
	
	
	addWeighted(reverse_res, 0.5, res, 0.5, 0.0, reverse_res);
	/*imshow("合成图片", reverse_res);*/
	reverse_res = Gussiblu(reverse_res,inputimage,reverse_res);
	/*imshow("合成图片2", reverse_res);*/
	outputimage = OldschoolStyle(reverse_res, outputimage);
	
	
	/*blur(outputimage, outputimage, Size(2.5, 2.5), Point(-1, -1), BORDER_DEFAULT);*/

	return outputimage;
}

int main()
{
	// 读入一张图片    
	Mat img1 = imread("test2.png");//也可以用"C:\\Users\\学习者\\Desktop\\lomo\\test.png"里的经典照片
	
	// 在窗口中显示图片   
	/*imshow("原始图片", img1);*/

	Mat Resiseimg1;
	if (img1.cols > 640)
	{
		resize(img1, Resiseimg1, Size(640, 640 * img1.rows / img1.cols));
		imshow("尺寸转换图", Resiseimg1);
		Mat new_img1 = Mat::zeros(Resiseimg1.size(), Resiseimg1.type());//创造一张与原图像大小和类型相同的空白图像，像素值初始化为0
		new_img1 = OldschoolStyle(Resiseimg1, new_img1);
		
		Mat lomo = Mat::zeros(Resiseimg1.size(), Resiseimg1.type());//创造一张与原图像大小和类型相同的空白图像，像素值初始化为0
		lomo = RoiReverse(Resiseimg1,lomo);
		imshow("lomo滤镜图", lomo);
		imwrite("after_lomo.png",lomo);

	
	}
	else
	{
		Mat new_img1 = Mat::zeros(img1.size(), img1.type());//创造一张与原图像大小和类型相同的空白图像，像素值初始化为0
		new_img1 = OldschoolStyle(img1, new_img1);

		Mat lomo = Mat::zeros(Resiseimg1.size(), Resiseimg1.type());//创造一张与原图像大小和类型相同的空白图像，像素值初始化为0
		lomo = RoiReverse(Resiseimg1, lomo);
		imshow("lomo滤镜图", lomo);
	
		imshow("怀旧风格", new_img1);
		imwrite("after_lomo.png", lomo);
	}
	

	
	// 等待50000ms后窗口自动关闭    
	waitKey(50000);
	



}