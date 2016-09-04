
#include <opencv2/highgui/highgui.hpp>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
vector<vector<Point> > polygons;
vector<Vec4i> hierarchy;

int main()
{
    /*Mat im = imread("C:/Users/SHANKAR1/Desktop/ip/lena.jpg");
    imshow("Window name",im);
    (Mat x;
    if(im.empty))
    {
        cout<<"Cannot load image!"<<endl;
        return -1;
    }
    cvtColor(im,x,CV_BGR2GRAY);

    imshow("Gray", x);
    waitKey(0);*/
    VideoCapture cam(0);
    if(cam.isOpened()==false)
    {
        cout<<"camera not found"<<endl;
        return 0;
    }
    Mat orig;
    int t=0;
    namedWindow("Trackbar");
    createTrackbar("Threshold","Trackbar",&t,255);
    vector<vector<Point> > contours;
    while(1)
    {
        if(!cam.read(orig))
            return 0;
        //imshow("Frame",orig);
        Mat x;
        cvtColor(orig,x,CV_BGR2GRAY);
        //imshow("gray",x);
        threshold(x,x,t,255,CV_THRESH_BINARY);
        findContours(x.clone(),contours,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
        for(int i=0;i<contours.size();i++)
        {
            drawContours(orig,contours,i,Scalar(255,0,0),2);
        }
        for (int i = 0; i < contours.size(); i++)
                            {
                                approxPolyDP(Mat(contours[i]), polygons[i], arcLength(Mat(contours[i]), true)*0.019, true);
                            }
        polygons.resize(contours.size());
                            for (int i = 0; i < polygons.size(); i++)
                            {
                                Scalar color = Scalar(0, 255, 255);

                                drawContours(orig, polygons, i, color, 3, 8, hierarchy, 0, Point());

                           }
        imshow("Frame1",orig);
        imshow("Gray1",x);
        if(waitKey(10)==27)
            return 0;

    }
    return 0;
}
