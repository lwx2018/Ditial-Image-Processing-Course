
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>


using namespace cv;
using namespace std;



Mat myBilateralFilter(Mat inputimage, Mat outputimage)
{
	/*Mat channel[3];
	split(inputimage, channel);*/
	int channels = inputimage.channels();
	int n = 25;//滤波核半径大小，它越大越平越模糊	
	int size = 2 * n + 1;
	double sigmas = 25;//用于计算空间权值的参数
	double sigmar = 25;//用于计算相似度权值的参数
	double **space_array = new double*[size + 1];//初始化空间权值数组,最后一行放总值
	for (int i = 0; i < size + 1; i++)
	{
		space_array[i] = new double[size + 1];
	}
	int rx = 0;
	int ry = 0;
	space_array[size][0] = 0.0f;
	
	double sum = 0;
	for (int i = 0; i < size; i++)//计算空间权值
	{
		for (int j = 0; j < size; j++)
		{
			rx = i - size / 2;
			ry = j - size / 2;
			int a = rx ^ 2;
			int b = ry ^ 2;

			space_array[i][j] = exp(((-0.5f) * (a + b)) / (sigmas * sigmas));//高斯函数
			space_array[size][0] = space_array[size][0] + space_array[i][j];
			

		}
	}

	double *color_array = new double[255 * channels + 1];//初始化相似度权值数组,最后一位放总值
	double wr = 0.0f;
	color_array[255 * channels + 1] = 0.0f;
	for (n = 0; n < 255 * channels + 1; n++) //计算相似度权值
	{
		color_array[n] = exp((-1.0f*(n*n)) / (2.0f*sigmar*sigmar));
		color_array[255 * channels + 1] += color_array[n];
	}
	
	for (int i = 0; i < inputimage.rows; i++)//开始滤波计算
	{
		for (int j = 0; j < inputimage.cols; j++)
		{
			if (i > size / 2 - 1 && j > size / 2 - 1 && i < inputimage.rows - size / 2 && j < inputimage.cols - size / 2)
			{
				double sum[3] = { 0.0,0.0,0.0 };
				int x, y, values;
				double scsum = 0.0f;
				for (int k = 0; k < size; k++)
				{
					for (int l = 0; l < size; l++)
					{
						x = i - k + size / 2;//原图的（x，y）是输入点，（i，j）是当前输出点
						y = j - l + size / 2;
						values = abs(inputimage.at<Vec3b>(i, j)[0] + inputimage.at<Vec3b>(i, j)[1] + inputimage.at<Vec3b>(i, j)[2] - inputimage.at<Vec3b>(x, y)[0] - inputimage.at<Vec3b>(x, y)[1] - inputimage.at<Vec3b>(x, y)[2]);
						for (int m = 0; m < 3; m++)
						{
							sum[m] += (inputimage.at<Vec3b>(x, y)[m] * space_array[k][l] * color_array[values]);
						}
					}
				}
				for (int m = 0; m < 3; m++)
				{
					outputimage.at<Vec3b>(i, j)[m] = sum[m];
				}
			}
		}
	}
	


	return outputimage;
}

Mat Gauss(Mat inputimage, double **array, int size)//高斯滤波操作
{

	Mat newimg = inputimage.clone();
	for (int i = 0; i < inputimage.rows; i++)
	{
		for (int j = 0; j < inputimage.cols; j++)
		{
			if (i > size / 2 - 1 && j > size / 2 - 1 && i < inputimage.rows - size / 2 && j < inputimage.cols - size / 2)
			{
				double sum = 0.0;
				for (int k = 0; k < size; k++)
				{
					for (int l = 0; l < size; l++)
					{
						sum += inputimage.ptr<uchar>(i - k + (size / 2))[j - l + (size / 2)] * array[k][l];
						
					}
					
				}
				newimg.ptr<uchar>(i)[j] = sum;
			}
		}
	}
	inputimage = newimg.clone();
	return inputimage;
}

Mat GaussFilter(Mat inputimage, Mat outputimage)//高斯滤波操作包括图像分离通道，确定卷积核
{
	Mat channel[3];
	split(inputimage, channel);//彩色图片通道分离
	/*imshow("B通道图像", channel[0]);
	imshow("G通道图像", channel[1]);
	imshow("R通道图像", channel[2]);*/
	int arr_size = 3;//卷积核大小
	double **array = new double*[arr_size];//初始化卷积矩阵
	for (int i = 0;i < arr_size;i++)
	{
		array[i] = new double[arr_size];
	}
	int rx = 0;
	int ry = 0;
	int sigma = 1.6;
	int c = sigma ^ 2;
	double sum = 0;
	for (int i = 0; i < arr_size; i++)
	{
		for (int j = 0; j < arr_size; j++)
		{
			rx = i - arr_size / 2;
			ry = j - arr_size / 2;
			int a = rx ^ 2;
			int b = ry ^ 2;

			array[i][j] =  exp(((-0.5) * (a + b)) / c);//归一化的二维高斯函数
			sum = array[i][j] + sum;

		}
	}
	for (int i = 0; i < arr_size; i++) //计算权值后的矩阵
	{
		for (int j = 0; j < arr_size; j++) 
		{
			array[i][j] = array[i][j] / sum;
		}
	}
	for (int v = 0; v < 3; v++) 
	{
		channel[v] = Gauss(channel[v], array, arr_size);
		/*for (int i = 0; i < channel[v].rows; i++) 
		{
			for (int j = 0; j < channel[v].cols; j++)
			{
				if (i > arr_size / 2 - 1 && j > arr_size / 2 - 1 && i < channel[v].rows - arr_size / 2 && j < channel[v].cols - arr_size / 2)
				{
					double sum = 0.0;
					for (int k = 0; k < arr_size; k++)
					{
						for (int l = 0; l < arr_size; l++)
						{
							sum += channel[v].ptr<uchar>(i - k + (arr_size / 2))[j - l + (arr_size / 2)] * array[k][l];
						}
						outputimage.ptr<uchar>(i)[j] = sum;
					}
				}
			}
		}*/
		
	}
	merge(channel,3 ,outputimage);
	return outputimage;

}


Mat RoiReverse(Mat inputimage, Mat outputimage) {

	Mat roi, res, reverse_res, reverse_roi;
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
	//画矩形roi区域
	int thickness = -1;//表示画的是实心的，非负数即是空心的
	int linetype = 8;
	
	rectangle(roi, Point(180, 180), Point(392, 392), Scalar(255, 255, 255), thickness, linetype);//黑色图像画白线，即黑底白线矩形
	/*imshow("roi——rectangle", roi);*/
	rectangle(reverse_roi, Point(180, 180), Point(392, 392), Scalar(0, 0, 0), thickness, linetype);//白色图像画黑线，即白底黑线矩形
	/*imshow("reverse_roi", reverse_roi);*/
	

	inputimage.copyTo(res, roi);
	/*imshow("取部分关注的地方",res);*/
	inputimage.copyTo(reverse_res, reverse_roi);
	/*imshow("取部分不关注的地方", reverse_res);*/
	res = GaussFilter(res, res);
	/*imshow("Gauss滤波", res);*/
	addWeighted(reverse_res, 0.5, res, 0.5, 0.0, reverse_res);
	/*imshow("合成图片", reverse_res);*/
	
	
	for (int row = 0; row < reverse_res.rows; row++)//改变reverse_res图像的亮度和对比度使用两个参数来改变分别是阿尔法和贝塔，有时分别控制对比度和亮度，阿尔法必大于零
	{
		for (int col = 0; col < reverse_res.cols; col++)
		{
			double alpha = 2.5;
			double beta = 5;

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
	rectangle(reverse_res, Point(180, 180), Point(392, 392), Scalar(0, 0, 255), 3, linetype);
	/*imshow("转换结果2", reverse_res);*/

	outputimage = reverse_res;
	

	return outputimage;
}
int main()
{

	// 读入一张图片    
	Mat img1 = imread("test.png");

	// 在窗口中显示图片   
	imshow("原始图片", img1);
	
	Mat skin = Mat::zeros(img1.size(), img1.type());

	skin = RoiReverse(img1, skin);
	imshow("美肤图像", skin);
	imwrite("skin.png", skin);


	// 等待50000ms后窗口自动关闭    
	waitKey(50000);
}


