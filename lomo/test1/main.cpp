#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>
#include <math.h>


using namespace cv;
using namespace std;


Mat OldschoolStyle(Mat inputimage,Mat outputimage) //ת��ͼ����תΪ���ɷ��
{
	
	for (int i = 0; i < inputimage.rows; i++)//.rows�Ƕ����ͼƬ��������ֵ
	{
		for (int j = 0; j < inputimage.cols; j++) //.cols�Ƕ����ͼƬ��������ֵ
		{
			outputimage.at<Vec3b>(i, j)[0] = 10* sqrt(inputimage.at<Vec3b>(i, j)[0]);//��ͼ����ɫͨ��ֵ�ع�
			outputimage.at<Vec3b>(i, j)[1] = (inputimage.at<Vec3b>(i, j)[1]);//��ͼ����ɫͨ��ֵ�ع�
			outputimage.at<Vec3b>(i, j)[2] = (inputimage.at<Vec3b>(i, j)[2]);//��ͼ���ɫͨ��ֵ�ع�
		}

	}
	return outputimage;
}


Mat Gussiblu(Mat inputimage1, Mat inputimage2, Mat outputimage)//�˲���ʵ��
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
				outputimage.at<Vec3b>(i, j)[0] = inputimage1.at<Vec3b>(i, j)[0] * w + inputimage2.at<Vec3b>(i, j)[0] * (1 - w);//��ͼ����ɫͨ��ֵ�ع�
				outputimage.at<Vec3b>(i, j)[1] = inputimage1.at<Vec3b>(i, j)[1] * w + inputimage2.at<Vec3b>(i, j)[1] * (1 - w);//��ͼ����ɫͨ��ֵ�ع�
				outputimage.at<Vec3b>(i, j)[2] = inputimage1.at<Vec3b>(i, j)[2] * w + inputimage2.at<Vec3b>(i, j)[2] * (1 - w);//��ͼ���ɫͨ��ֵ�ع�
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
	reverse_roi = Mat::zeros(inputimage.size(), inputimage.type());//����һ����ԭͼ���С��������ͬ�Ŀհ�ͼ������ֵ��ʼ��Ϊ0����Ϊ��ɫͼ��
	for (int i = 0; i < reverse_roi.rows; i++)//.rows�Ƕ����ͼƬ��������ֵ������ͼ���Ϊ��ɫ
	{
		for (int j = 0; j < reverse_roi.cols; j++) //.cols�Ƕ����ͼƬ��������ֵ
		{
			reverse_roi.at<Vec3b>(i, j)[0] = 255;//��ͼ����ɫͨ��ֵ�ع�
			reverse_roi.at<Vec3b>(i, j)[1] = 255;//��ͼ����ɫͨ��ֵ�ع�
			reverse_roi.at<Vec3b>(i, j)[2] = 255;//��ͼ���ɫͨ��ֵ�ع�
		}

	}
	/*imshow("roi",roi);
	imshow("reverse_roi", reverse_roi);*/
	//����Բroi����
	int thickness = 2;
	int linetype = 8;
	double angle = 90;
	double w = inputimage.cols;
	double v = inputimage.rows;
	ellipse(roi, Point(w / 2.0, 2 * v / 3.0), Size(2 * v /3.0, v / 3.0), angle, 0, 360, Scalar(255, 255, 255), thickness, linetype);//��ɫͼ�񻭰��ߣ����ڵװ�����Բ
	/*imshow("roi����ellipse", roi);*/
	ellipse(reverse_roi, Point(w / 2.0, 2 * v / 3.0), Size(2 * v / 3.0, v / 3.0), angle, 0, 360, Scalar(0.0,0), thickness, linetype);//��ɫͼ�񻭺��ߣ����׵׺�����Բ
	/*imshow("reverse_roi", reverse_roi);*/
	floodFill(reverse_roi, Point(w / 2.0, v / 2.0), Scalar(0, 0, 0));//���һ����Բ�����ڵ���ɫΪ��ɫ����ʱͼ��Ϊ�׵׺�ɫʵ����Բ
	/*imshow("reverse_roi", reverse_roi);*/
	floodFill(roi, Point(w / 2.0, v / 2.0), Scalar(255, 255, 255));//���һ����Բ�����ڵ���ɫΪ��ɫ����Ϊ�ڵװ�ʵ����Բ
	/*imshow("roi", roi);*/
	
	inputimage.copyTo(res, roi);
	/*imshow("ȡ���ֹ�ע�ĵط�",res);*/
	inputimage.copyTo(reverse_res, reverse_roi);
	/*imshow("ȡ���ֲ���ע�ĵط�", reverse_res);*/
	/*cvtColor(res, res, CV_BGR2HSV);*/
	/*imshow("ת��ɫ��", res);*/
	for (int row = 0; row < res.rows; row++)//�ı�resͼ������ȺͶԱȶ�ʹ�������������ı�ֱ��ǰ������ͱ�������ʱ�ֱ���ƶԱȶȺ����ȣ��������ش�����
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
	

	/*imshow("ת�����1", res);*/
	/*cvtColor(reverse_res, reverse_res, CV_BGR2HSV);*/
	/*imshow("ת��ɫ��", reverse_res);*/
	for (int row = 0; row < reverse_res.rows; row++)//�ı�reverse_resͼ������ȺͶԱȶ�ʹ�������������ı�ֱ��ǰ������ͱ�������ʱ�ֱ���ƶԱȶȺ����ȣ��������ش�����
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
	/*imshow("ת�����2", reverse_res);*/
	
	
	addWeighted(reverse_res, 0.5, res, 0.5, 0.0, reverse_res);
	/*imshow("�ϳ�ͼƬ", reverse_res);*/
	reverse_res = Gussiblu(reverse_res,inputimage,reverse_res);
	/*imshow("�ϳ�ͼƬ2", reverse_res);*/
	outputimage = OldschoolStyle(reverse_res, outputimage);
	
	
	/*blur(outputimage, outputimage, Size(2.5, 2.5), Point(-1, -1), BORDER_DEFAULT);*/

	return outputimage;
}

int main()
{
	// ����һ��ͼƬ    
	Mat img1 = imread("test2.png");//Ҳ������"C:\\Users\\ѧϰ��\\Desktop\\lomo\\test.png"��ľ�����Ƭ
	
	// �ڴ�������ʾͼƬ   
	/*imshow("ԭʼͼƬ", img1);*/

	Mat Resiseimg1;
	if (img1.cols > 640)
	{
		resize(img1, Resiseimg1, Size(640, 640 * img1.rows / img1.cols));
		imshow("�ߴ�ת��ͼ", Resiseimg1);
		Mat new_img1 = Mat::zeros(Resiseimg1.size(), Resiseimg1.type());//����һ����ԭͼ���С��������ͬ�Ŀհ�ͼ������ֵ��ʼ��Ϊ0
		new_img1 = OldschoolStyle(Resiseimg1, new_img1);
		
		Mat lomo = Mat::zeros(Resiseimg1.size(), Resiseimg1.type());//����һ����ԭͼ���С��������ͬ�Ŀհ�ͼ������ֵ��ʼ��Ϊ0
		lomo = RoiReverse(Resiseimg1,lomo);
		imshow("lomo�˾�ͼ", lomo);
		imwrite("after_lomo.png",lomo);

	
	}
	else
	{
		Mat new_img1 = Mat::zeros(img1.size(), img1.type());//����һ����ԭͼ���С��������ͬ�Ŀհ�ͼ������ֵ��ʼ��Ϊ0
		new_img1 = OldschoolStyle(img1, new_img1);

		Mat lomo = Mat::zeros(Resiseimg1.size(), Resiseimg1.type());//����һ����ԭͼ���С��������ͬ�Ŀհ�ͼ������ֵ��ʼ��Ϊ0
		lomo = RoiReverse(Resiseimg1, lomo);
		imshow("lomo�˾�ͼ", lomo);
	
		imshow("���ɷ��", new_img1);
		imwrite("after_lomo.png", lomo);
	}
	

	
	// �ȴ�50000ms�󴰿��Զ��ر�    
	waitKey(50000);
	



}