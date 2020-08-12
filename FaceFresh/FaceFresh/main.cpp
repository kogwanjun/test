/*#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, copyimg, img_th, inpaint_img_result, denosing_img_result, img_gy;
bool mousedown;
vector<vector<Point> > contours;
vector<Point> pts;
//void load_cascade(CascadeClassifier& cascade, string fname)
//{
//	String path = "C:\\opencv410\\opencv\\sources\\data\\haarcascades\\";
//	String full_name = path + fname;
//
//	CV_Assert(cascade.load(full_name));
//}
bool findPimples(Mat img)
{
	Mat bw, bgr[3];
	split(img, bgr);
	bw = bgr[1];
	int pimplescount = 0;
	vector<Rect> blurrect;
	adaptiveThreshold(bw, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
	dilate(bw, bw, Mat(), Point(-1, -1), 1);

	contours.clear();
	findContours(bw, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > 25 & contourArea(contours[i]) < 120)//25,120
		{
			Rect im_rect, minRect = boundingRect(Mat(contours[i]));

			Mat imgroi(img, minRect);

			cvtColor(imgroi, imgroi, COLOR_BGR2HSV);
			Scalar color = mean(imgroi);
			cvtColor(imgroi, imgroi, COLOR_HSV2BGR);

			if (color[0] < 30 & color[1] > 90 & color[2] > 50)// 30,90,50
			{
				Point2f center;
				float radius = 0;
				minEnclosingCircle(Mat(contours[i]), center, radius);

				if (radius < 50)//50
				{
					rectangle(img, minRect, Scalar(0, 0, 0), -1);
					pimplescount++;
				}
				blurrect.push_back(minRect);
			}
		}

	}
	//githup에서 수정하였습니다.
	//putText(img, format("%d", pimplescount), Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
	imshow("pimples dedector", img);//copyimg 원본이미지, img 여드름체킹이미지, img_th 이진화이미지, img_result 결과 이미지

	cvtColor(img, img_gy, COLOR_BGR2GRAY);
	threshold(img_gy, img_th, 0, 255, 1);

	inpaint(img, img_th, inpaint_img_result, 15, INPAINT_TELEA); //circular neighborhood의 radius, 15
	fastNlMeansDenoisingColored(inpaint_img_result, denosing_img_result, 3, 3, 7, 21);//3,3,7,21
	imshow("denosing_img_result", denosing_img_result);
	//imshow("img_th", img_th);

	//imshow("inpaint_img_result", inpaint_img_result);
	return 0;
}


int main()
{
	src = imread("pimples.jpg", 1);

	if (src.empty())
	{
		return -1;
	}
	copyimg = src.clone();

	imshow("원본이미지", copyimg);
	findPimples(src);

	waitKey(0);
	return 0;
}*/
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

void load_cascade(CascadeClassifier& cascade, string fname)
{
	//String path = "C:\\opencv410\\opencv\\sources\\data\\haarcascades\\";
	String path = "C:\\opencv349\\opencv\\sources\\data\\haarcascades\\";	
	String full_name = path + fname;

	CV_Assert(cascade.load(full_name));
}
Mat preprocessing(Mat image)
{
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);

	return gray;
}
Point2d calc_center(Rect obj)
{
	Point2d c = (Point2d)obj.size() / 2.0;
	Point2d center = (Point2d)obj.tl() + c;
	return center;
}
void detect_hair(Point2d face_center, Rect face, vector<Rect>& hair_rect)
{
	Point2d h_gap(face.width*0.45, face.height*0.65);
	Point2d pt1 = face_center - h_gap;
	Point2d pt2 = face_center + h_gap;
	Rect hair(pt1, pt2);

	Size size(hair.width, hair.height*0.40);
	Rect hair1(hair.tl(), size);
	Rect hair2(hair.br() - (Point)size, size);

	hair_rect.push_back(hair1);
	hair_rect.push_back(hair2);
	hair_rect.push_back(hair);
}
Mat calc_rotMap(Point2d face_center, vector<Point2d> pt)
{
	Point2d delta = (pt[0].x > pt[1].x) ? pt[0] - pt[1] : pt[1] - pt[0];
	double angle = fastAtan2(delta.y, delta.x);

	Mat rot_mat = getRotationMatrix2D(face_center, angle, 1);
	return rot_mat;
}
Mat src, copyimg, img_th, inpaint_img_result, denosing_img_result, img_gy;
vector<vector<Point> > contours;
vector<Point> pts;
bool findPimples(Mat img)
{
	Mat bw, bgr[3];
	split(img, bgr);
	bw = bgr[1];
	
	adaptiveThreshold(bw, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
	dilate(bw, bw, Mat(), Point(-1, -1), 1);

	contours.clear();
	findContours(bw, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > 25 & contourArea(contours[i]) < 120)//25,120
		{
			Rect im_rect, minRect = boundingRect(Mat(contours[i]));

			Mat imgroi(img, minRect);

			cvtColor(imgroi, imgroi, COLOR_BGR2HSV);
			Scalar color = mean(imgroi);
			cvtColor(imgroi, imgroi, COLOR_HSV2BGR);

			if (color[0] < 30 & color[1] > 90 & color[2] > 50)// 30,90,50
			{
				Point2f center;
				float radius = 0;
				minEnclosingCircle(Mat(contours[i]), center, radius);

				if (radius < 50)//50
					rectangle(img, minRect, Scalar(0, 0, 0), -1);
				
			}
		}

	}

	//putText(img, format("%d", pimplescount), Point(50, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
	imshow("pimples dedector", img);//copyimg 원본이미지, img 여드름체킹이미지, img_th 이진화이미지, img_result 결과 이미지

	cvtColor(img, img_gy, COLOR_BGR2GRAY);
	threshold(img_gy, img_th, 0, 255, 1);

	inpaint(img, img_th, inpaint_img_result, 15, INPAINT_TELEA); //circular neighborhood의 radius, 15
	fastNlMeansDenoisingColored(inpaint_img_result, denosing_img_result, 3, 3, 7, 21);//3,3,7,21
	imshow("denosing_img_result", denosing_img_result);
	//imshow("img_th", img_th);

	//imshow("inpaint_img_result", inpaint_img_result);
	return 0;
}
void DarkCircle(Mat img, Mat & result_img) {
	//Mat img_gy,bw,bgr[3];
	////cvtColor(img, result_img, COLOR_BGR2HSV);
	//split(img, bgr);
	//bw = bgr[0];
	//adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 5);
	
	//imshow("bw", img);
	fastNlMeansDenoisingColored(img, result_img, 3, 3, 7, 21);
}
int main()
{
	CascadeClassifier face_cascade, eyes_cascade;
	load_cascade(face_cascade, "haarcascade_frontalface_alt2.xml");
	load_cascade(eyes_cascade, "haarcascade_eye.xml");

	Mat image = imread("pimples.jpg", 1);
	Mat second_image = imread("darkcircle4.jpg", 1);
	Mat second_result_image;
	copyimg = image.clone();//원본이미지 복사
	Mat gray = preprocessing(image);
	
	vector<Rect> faces, eyes;
	face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(100, 100));
	DarkCircle(second_image, second_result_image);
	imshow("second_result_image", second_result_image);
	if (faces.size() > 0)
	{
		//얼굴 검출==----------------------------------------------------------------
		eyes_cascade.detectMultiScale(gray(faces[0]), eyes, 1.15, 7, 0, Size(25, 20));
		
		Point2d face_center = calc_center(faces[0]);

		vector<Rect> sub_obj;
		detect_hair(face_center, faces[0], sub_obj);
		rectangle(image, sub_obj[2], Scalar(0, 0, 0), 1);

	    //특정부분 검출---------------------------------------------------------------
		Mat roi_image(image, sub_obj[2]);//roi 이미지
	    //---------------------------------------------------------------------------
		//imshow자리
		imshow("원본이미지", copyimg);
		imshow("face_detection_result", image);
		imshow("roi_result", roi_image);
		findPimples(roi_image);
		waitKey();
		
	}
	
	return 0;
}
/*int main()
{
	CascadeClassifier face_cascade, eyes_cascade;
	load_cascade(face_cascade, "haarcascade_frontalface_alt2.xml");
	load_cascade(eyes_cascade, "haarcascade_eye.xml");

	Mat second_image = imread("darkcircle7.jpg", 1);
	Mat second_result_image;
	Mat gray = preprocessing(second_image);
	copyimg = second_image.clone();

	vector<Rect> faces, eyes;
	face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(100, 100));
	//
	if (faces.size() > 0)
	{
		//얼굴 검출==----------------------------------------------------------------
		eyes_cascade.detectMultiScale(gray(faces[0]), eyes, 1.15, 7, 0, Size(25, 20));

		Point2d face_center = calc_center(faces[0]);

		vector<Rect> sub_obj;
		detect_hair(face_center, faces[0], sub_obj);
		rectangle(second_image, sub_obj[2], Scalar(0, 0, 0), 1);

		//특정부분 검출---------------------------------------------------------------
		Mat roi_image(second_image, sub_obj[2]);//roi 이미지
		//---------------------------------------------------------------------------
		DarkCircle(roi_image, second_result_image);
		//imshow자리
		//imshow("원본이미지", copyimg);
		//imshow("face_detection_result", second_image);
		//imshow("roi_result", roi_image);
		//imshow("second_result_image", second_result_image);
		waitKey();

	}



	return 0;
}*/
