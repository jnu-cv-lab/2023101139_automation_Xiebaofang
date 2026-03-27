#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    // 任务1: 读取图片
    cout << "任务1: 读取图片" << endl;

    // 读取图片
    Mat img = imread("images/test.jpg");
    
    // 检查图片是否读取成功
    if (img.empty())
    {
        cout << "错误: 无法读取图片，请检查图片路径和文件名" << endl;
        return -1;
    }
    
    cout << "图片读取成功!" << endl;

    // 任务2: 输出图像基本信息
    cout << "\n任务2: 输出图像基本信息" << endl;
    cout << "图像宽度: " << img.cols << " 像素" << endl;
    cout << "图像高度: " << img.rows << " 像素" << endl;
    cout << "图像通道数: " << img.channels() << endl;
    cout << "图像数据类型: " << typeToString(img.type()) << endl;

    // 任务3: 显示原图
    cout << "\n任务3: 显示原图" << endl;
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", img);

    // 任务4: 转换为灰度图并显示
    cout << "\n任务4: 转换为灰度图" << endl;
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray_img);

    // 任务5: 保存灰度图 → 保存到 images/ 文件夹
    cout << "\n任务5: 保存灰度图" << endl;
    imwrite("images/gray_test.jpg", gray_img);
    cout << "灰度图已保存到: images/gray_test.jpg" << endl;

    // 任务6: 简单像素操作 + 裁剪保存 → 保存到 images/ 文件夹
    cout << "\n任务6: 简单像素操作" << endl;
    // 输出左上角(0,0)像素值
    if (img.channels() == 3) {
        Vec3b pixel = img.at<Vec3b>(0, 0);
        cout << "左上角(0,0)像素值(BGR): " << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << endl;
    } else {
        uchar pixel = img.at<uchar>(0, 0);
        cout << "左上角(0,0)像素值: " << (int)pixel << endl;
    }
    // 裁剪左上角100x100区域
    Mat roi = img(Rect(0, 0, 100, 100));
    imwrite("images/roi_test.jpg", roi);
    cout << "裁剪图已保存到: images/roi_test.jpg" << endl;

    // 等待按键关闭窗口
    waitKey(0);
    destroyAllWindows();

    return 0;
}