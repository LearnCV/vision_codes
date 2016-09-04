#include <stdio.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//---------- All Global  Variables----------------------------------
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

vector<vector<Point> > polygons;
Point2f cent;
Point2f angle_base;
double dist;
double rotate_angle;

//--------------------------------------------------------------------------

void gray_world(Mat src1,float *ml,float *ma, float *mb, int p,int m)
{
        *ma=0;
        *mb=0;
        *ml=0;

        for(int i=0;i<src1.rows;i++)
                {
                        for(int j=0;j<src1.cols;j++)
                        {
                                Vec3b v1=src1.at<Vec3b>(i,j);
                                float lc=pow(v1.val[0],p);
                                float ac=pow(v1.val[1],p);
                float bc=pow(v1.val[2],p);
                *ma=*ma+ac;
                *mb=*mb+bc;
                *ml=*ml+lc;
            }
    }

    *ma=pow((float)*ma/(src1.cols*src1.rows),(float)1/p);
    *mb=pow((float)*mb/(src1.cols*src1.rows),(float)1/p);
    *ml=pow((float)*ml/(src1.cols*src1.rows),(float)1/p);


     float r=0;

     if(m==0)
     {
         r=(*ma+*mb+*ml)/3;
         *ma=r/(*ma);
         *mb=r/(*mb);
         *ml=r/(*ml);

     }

     if(m==1)
     {
         r=(*ma+*mb+*ml)/3;
         r=max(*ma,*mb);
         r=max(r,*ml);


         *ma=r/(*ma);
                 *mb=r/(*mb);
                 *ml=r/(*ml);


                }
                if(m==2)
                {
                    r=sqrt((*ma)*(*ma)+(*mb)*(*mb)+(*ml)*(*ml));
                   *ma=(float)(*ma)/(float)r;
                    *mb=(float)(*mb)/(float)r;
                    *ml=(float)(*ml)/(float)r;

                    cerr <<  *ml << endl;
                    cerr <<  *ma << endl;
                    cerr <<  *mb << endl;

                    r=max(*ma,*mb);
                    r=max(r,*ml);

                    *ma=(float)r/(float)(*ma);
                    *mb=(float)r/(float)(*mb);
                    *ml=(float)r/(float)(*ml);
                }
     }

Mat white_balance(Mat src,int p_in,int m_in)
{
    vector<Mat> bgr_planes;
    //Mat src = image;
    Mat dst;
    src.copyTo(dst);
    Mat src1 = src;
    //split(src,bgr_planes);
    split(src,bgr_planes);

    float ma=0,mb=0,ml=0;

           int p=p_in;
           int m=m_in;

           gray_world(src,&ml,&ma,&mb,p,m);


           float r=(ma+mb+ml)/3;
           if(m==1)
           {
               r=(ma+mb+ml)/3;
               r=max(ma,mb);
               r=max(r,ml);
           }

           MatIterator_<Vec3b> it=src1.begin<Vec3b>();
           MatIterator_<Vec3b> itend=src1.end<Vec3b>();
           MatIterator_<Vec3b> itout=dst.begin<Vec3b>();

           for (;it!=itend;++it,++itout)
           {
               Vec3b v1=*it;

               float l=v1.val[0];
               float a=v1.val[1];
               float b=v1.val[2];


               a=a*(r/ma);
               b=b*(r/mb);
               l=l*(r/ml);

               if(a>255)
                   a=255;
               if(b>255)
                   b=255;
               if(l>255)
                   l=255;
               v1.val[0]=l;
               v1.val[1]=a;
               v1.val[2]=b;
               *itout=v1;
           }
           return dst;
}

Mat clahe_conversion(Mat input)
{
    vector<Mat> RGB; // Use the STLâ€™s vector structure to store multiple Mat objects
    split(input, RGB); // split the image into separate color planes (R G B)
    ////    Enhance Local Contrast (CLAHE)
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    Mat RGB_eq;


    ////    Equalizes the histogram of a one channel image  (8UC1) using Contrast Limited Adaptive Histogram Equalization.
    clahe->apply(RGB[0],RGB[0]);
    clahe->apply(RGB[1],RGB[1]);
    clahe->apply(RGB[2],RGB[2]);

    merge(RGB,RGB_eq);              // now merge the results back
    return RGB_eq;
}

double abs_distance(Point2f p1, Point2f p2)
{
    double d = (p1.x - p2.x)*(p1.x - p2.x) +(p1.y - p2.y)*(p1.y - p2.y);
    double dist = pow(d, 0.5);
    return dist;
}

double cc_angle(Point2i vtx, Point2i p1, Point2i p2)
{
    // 1st point is the vertex , then 1st point and 2nd point,angle is obtained in radians
    double modv1 = abs_distance(vtx, p1);		// Mod v1
    double modv2 = abs_distance(vtx, p2);		// Mod v2
    double dot = (p1.x - vtx.x)*(p2.x - vtx.x) + (p1.y - vtx.y)*(p2.y - vtx.y);
    double costheta = dot / (modv1*modv2);
    double sign = 0;
    double term = (p1.x - vtx.x)*(p2.y - vtx.y) - (p1.y - vtx.y)*(p2.x - vtx.x);
    if (term <= 0)
        sign = 1.0;
    else
        sign = -1.0;
    double theta = 0;
    if (costheta >= 1.0)
        theta = 0;
    else if (costheta <= -1.0)
        theta = acos(-1.0);
    else
        theta = acos(costheta);
    return theta*sign;

}



int main(){

    VideoCapture cap("/home/shaggy/line_auv/GOPR0033.MP4");
    if(!cap.isOpened()){
        cout<< "Cannot open the video file" << endl;
        return -1;
    }

    namedWindow("Video",CV_WINDOW_NORMAL);

    while(1){
        Mat frame;

        bool bSuccess = cap.read(frame);

         if (!bSuccess){
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        Mat image = frame;
        Mat image_hsv;

        cent.y=frame.rows/2;
        cent.x=frame.cols/2;
        angle_base.y=frame.rows/2;
        angle_base.x=0;
        image = clahe_conversion(image);
        namedWindow("normal",CV_WINDOW_NORMAL);
        imshow("normal", frame);
        addWeighted(frame,0.7, image, 0.3, 0.0, image);
        //image = white_balance(image, 1,1);
        namedWindow("white_balance",CV_WINDOW_NORMAL);
        imshow("white_balance", image);

        cvtColor(image, image_hsv, COLOR_BGR2HSV);

        vector<Mat>hsv;
        split(image_hsv, hsv);
        imwrite("/home/harsha/Pictures/clahee.JPG",image);


        namedWindow("hsv",CV_WINDOW_NORMAL);

        namedWindow("h",CV_WINDOW_NORMAL);
        namedWindow("saturation",CV_WINDOW_NORMAL);
        imshow("hsv",image_hsv);

        imshow("h",hsv[0]);
        imshow("saturation", hsv[1]);


        Mat hsv_image = hsv[0];
        Mat hsv_image1 = hsv[1];

        #pragma omp parallel for collapse(2)
        for(int y=0; y<hsv_image.cols; y++){
            for(int x=0; x<hsv_image.rows; x++){
                if(hsv_image.at<uchar>(x,y) >=0 && hsv_image.at<uchar>(x,y) <25 || hsv_image.at<uchar>(x,y) >160 && hsv_image.at<uchar>(x,y)<180){
                    hsv_image.at<uchar>(x,y) = 255;
                }
                else
                    hsv_image.at<uchar>(x,y) = 0;
            }
        }
        #pragma omp parallel for collapse(2)
        for(int y=0; y<hsv_image1.cols; y++){
            for(int x=0; x<hsv_image1.rows; x++){
                if(hsv_image1.at<uchar>(x,y) >=0 && hsv_image1.at<uchar>(x,y) <40 ){
                    hsv_image.at<uchar>(x,y) = 0;
                }
                else
                    //hsv_image.at<uchar>(x,y);
                    continue;
            }
        }
      /*  #pragma omp parallel for collapse(2)
        for(int y=0; y<hsv_image.cols; y++){
            for(int x=0; x<hsv_image.rows; x++){
                if(hsv_image1.at<uchar>(x,y) == 0){
                    hsv_image.at<uchar>(x,y) = 0;
                }
                else
                    hsv_image.at<uchar>(x,y);
            }
        }*/

        //cvtColor(hsv_image,hsv_image,CV_BGR2GRAY);
        Mat elm = getStructuringElement(MORPH_RECT, Size(17, 17));

        erode(hsv_image,hsv_image, elm);
        erode(hsv_image,hsv_image, elm);
        dilate(hsv_image,hsv_image, elm);
        //erode(src_gray, src_gray, elm);
        //dilate(src_gray, src_gray, elm);

        findContours(hsv_image.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

                int aHighIndex=0;
                    double aHigh = 0;
                    //cout<<contours.size()<<endl;
                    for( int i = 0; i< contours.size(); i++ )
                    {
                        if(contourArea(contours[i]) > aHigh)
                        {
                            aHighIndex = i;
                            aHigh = contourArea(contours[i]);
                        }
                    }
        cout<<aHigh<<endl;
        Mat path = Mat::zeros(hsv_image.size(),CV_8UC3);
        drawContours(path,contours,aHighIndex,Scalar::all(255),2, 8, hierarchy, 0, Point());
        Moments mc_path;
        Point2f path_centre;
        circle(path,cent,40,Scalar(0,255,0),5,8,0);
        if(contours.size()!=0)
        {
            mc_path=moments(contours[aHighIndex],true);

            path_centre=Point2f(mc_path.m10/mc_path.m00,mc_path.m01/mc_path.m00);
            circle(path,path_centre,40,Scalar(255,0,0),5,8,0);
            dist=abs_distance(path_centre,cent);
            line(path,path_centre,cent,Scalar(255,255,0),1,8,0);
            rotate_angle= cc_angle(cent,path_centre,angle_base);
            cout<<"angle"<<rotate_angle<<endl;


        }
            cout<<path_centre<<endl;
        //drawing, contours, i, color, 2, 8, hierarchy, 0, Point()
        namedWindow("contours",CV_WINDOW_NORMAL);


        namedWindow("saturation1",CV_WINDOW_NORMAL);
        //namedWindow("path",CV_WINDOW_NORMAL);
        //imshow("saturation1",hsv_image1);
        imshow("path",hsv_image);
        imshow("contours",path);

        //imshow("MyVideo", frame);

        if(waitKey(30) == 27)
       {
                cout << "esc key is pressed by user" << endl;
                break;
       }

    }

}
