// 诊断：溢料判据在 0110 上检出了什么
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif
#include "chardet.hpp"
#include "common/console_utf8.hpp"
namespace fs=std::filesystem;
namespace{
std::string w2u(const std::wstring& w){if(w.empty())return{};int n=::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),nullptr,0,nullptr,nullptr);std::string o(n,'\0');::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),o.data(),n,nullptr,nullptr);return o;}
fs::path u8p(const std::string& s){return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
bool rd(const std::string& p,cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read((char*)b.data(),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<3)return 2;
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    cv::Mat g; if(!rd(a[2],g))return 1;
    cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
    if(!cm.ref.empty()){cv::Mat r32,s32;cm.ref.convertTo(r32,CV_32F);sig.convertTo(s32,CV_32F);auto sh=cv::phaseCorrelate(s32,r32,cv::noArray());if(std::abs(sh.x)<60&&std::abs(sh.y)<60){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}}

    cv::Mat tBin,mBin;
    cv::compare(sig,cm.config.strokeHi,tBin,cv::CMP_GT);
    cv::compare(cm.freq,static_cast<int>(cm.config.blankFreq*255),mBin,cv::CMP_GT);
    cv::Mat notM; cv::bitwise_not(mBin,notM);
    cv::Mat extra; cv::bitwise_and(tBin,notM,extra);
    { const cv::Mat kDil=cv::getStructuringElement(cv::MORPH_ELLIPSE,{9,9});
      cv::Mat inkDil,notNear; cv::dilate(mBin,inkDil,kDil); cv::bitwise_not(inkDil,notNear);
      cv::bitwise_and(extra,notNear,extra); }
    cv::Mat k3=cv::getStructuringElement(cv::MORPH_ELLIPSE,{3,3});
    cv::morphologyEx(extra,extra,cv::MORPH_OPEN,k3);

    cv::Mat labels,stats,cents;
    int n=cv::connectedComponentsWithStats(extra,labels,stats,cents,8,CV_32S);
    std::printf("溢料候选（面积80~3000 且 长宽比>2.2 且宽>=4）:\n");
    int found=0;
    for(int i=1;i<n;++i){
        int area=stats.at<int>(i,cv::CC_STAT_AREA);
        if(area<80||area>3000)continue;
        int x=stats.at<int>(i,cv::CC_STAT_LEFT),y=stats.at<int>(i,cv::CC_STAT_TOP);
        int w=stats.at<int>(i,cv::CC_STAT_WIDTH),h=stats.at<int>(i,cv::CC_STAT_HEIGHT);
        std::vector<cv::Point> pts;
        for(int yy=y;yy<y+h;++yy){const int* row=labels.ptr<int>(yy);for(int xx=x;xx<x+w;++xx)if(row[xx]==i)pts.emplace_back(xx,yy);}
        if(pts.size()<80)continue;
        cv::Mat data((int)pts.size(),2,CV_32F);
        for(size_t j=0;j<pts.size();++j){data.at<float>((int)j,0)=(float)pts[j].x;data.at<float>((int)j,1)=(float)pts[j].y;}
        cv::Mat mean,covar,evals;
        cv::calcCovarMatrix(data,covar,mean,cv::COVAR_NORMAL|cv::COVAR_ROWS);
        cv::eigen(covar,evals);
        if(evals.rows<2)continue;
        double e=std::sqrt(std::max(evals.at<double>(0),1e-6)/std::max(evals.at<double>(1),1e-6));
        int wd=std::min(w,h);
        if(e<=2.2)continue;
        if(wd<4)continue;
        std::printf("  (%d,%d) %dx%d 面积%d 长宽比%.1f\n",x,y,w,h,area,e);
        ++found;
    }
    std::printf("共 %d 个\n",found);
    return 0;
}