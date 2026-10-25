// 流程详解图生成器：从 tif 到检测结果
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
cv::Mat toBgr(const cv::Mat& g){cv::Mat o; if(g.channels()==1)cv::cvtColor(g,o,cv::COLOR_GRAY2BGR); else o=g.clone(); return o;}
cv::Mat cell(const cv::Mat& img,const std::string& title,int W,int H){ cv::Mat b=toBgr(img); double s=std::min((double)(W-8)/(double)b.cols,(double)(H-30)/(double)b.rows); cv::Mat r; cv::resize(b,r,cv::Size(),s,s,s>1?cv::INTER_NEAREST:cv::INTER_AREA); cv::Mat out=cv::Mat::zeros(H,W,CV_8UC3); r.copyTo(out(cv::Rect(4,26,r.cols,r.rows))); cv::putText(out,title,cv::Point(6,18),cv::FONT_HERSHEY_SIMPLEX,0.45,cv::Scalar(0,255,255),1,cv::LINE_AA); return out; }
cv::Mat labelled(const cv::Mat& img, const std::string& title, const cv::Scalar& col, int w=380){
    cv::Mat b=toBgr(img);
    double s=std::min((double)w/b.cols,(double)300/b.rows);
    cv::Mat r; cv::resize(b,r,cv::Size(),s,s,s>1?cv::INTER_NEAREST:cv::INTER_AREA);
    cv::copyMakeBorder(r,r,26,0,0,0,cv::BORDER_CONSTANT,cv::Scalar(30,30,30));
    cv::putText(r,title,cv::Point(6,18),cv::FONT_HERSHEY_SIMPLEX,0.45,col,1,cv::LINE_AA);
    return r;
}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<4){std::printf("用法: markloc_flow <模型目录> <原图tif> <输出png>\n");return 2;}
    chardet::Model cm; if(!chardet::loadModel(a[1],cm)){std::printf("model fail\n");return 1;}
    cv::Mat gray; if(!rd(a[2],gray)){std::printf("img fail\n");return 1;}
    chardet::Config cfg=cm.config;

    // 步骤1：原图
    cv::Mat s1=gray;

    // 步骤2：只取 v00 视角（本图就是），裁到塑封体
    cv::Rect body(300,225,1065,915);
    cv::Mat s2=gray(body).clone();

    // 步骤3：笔画信号（去背景+归一化）
    cv::Mat sig=chardet::strokeSignal(gray,cv::Rect(0,0,gray.cols,gray.rows),cfg,false);
    // 转成可视灰度
    cv::Mat sigVis;
    sig.convertTo(sigVis,CV_32F);
    cv::normalize(sigVis,sigVis,0,255,cv::NORM_MINMAX);
    sigVis.convertTo(sigVis,CV_8U);
    cv::Mat s3=sigVis(body).clone();

    // 步骤4：二值化（刻度信号）
    cv::Mat bin; cv::compare(sig,cfg.strokeHi,bin,cv::CMP_GT);
    cv::Mat s4=bin(body).clone();

    // 步骤5：模型频率图
    cv::Mat s5=cm.freq(body).clone();

    // 步骤6：字符框
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}
    cv::Mat s6; cv::cvtColor(gray,s6,cv::COLOR_GRAY2BGR);
    for(auto&u:all){ cv::rectangle(s6,u.bbox,cv::Scalar(0,255,0),2); }
    for(size_t ri=0;ri<cm.regions.size();++ri){ cv::rectangle(s6,cm.regions[ri].rect,cv::Scalar(255,0,0),2); }
    cv::Mat s6c=s6(body).clone();

    // 步骤7：逐字符对齐后
    cv::Mat s7; cv::Mat sigAligned=sig.clone();
    for(auto&u:all){
        cv::Rect box=u.bbox&cv::Rect(0,0,sig.cols,sig.rows);
        if(box.width<=4||box.height<=4)continue;
        constexpr int kS=12;
        cv::Rect search(box.x-kS,box.y-kS,box.width+2*kS,box.height+2*kS);
        search&=cv::Rect(0,0,sig.cols,sig.rows);
        if(search.width<box.width||search.height<box.height)continue;
        cv::Mat mS,tS; cm.freq(search).convertTo(mS,CV_32F,1.0/255.0); sig(search).convertTo(tS,CV_32F);
        cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-search.x,box.y-search.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
        cv::Point mLoc; double mVal=0; cv::minMaxLoc(res,nullptr,&mVal,nullptr,&mLoc);
        if(mVal<0.30)continue;
        u.bbox=cv::Rect(box.x+mLoc.x-kS,box.y+mLoc.y-kS,box.width,box.height);
    }
    cv::cvtColor(gray,s7,cv::COLOR_GRAY2BGR);
    for(auto&u:all){ cv::rectangle(s7,u.bbox,cv::Scalar(0,255,0),2); }
    cv::Mat s7c=s7(body).clone();

    // 拼图
    const int W=380,H=300;
    cv::Mat row1,row2;
    cv::hconcat(std::vector<cv::Mat>{cell(s1,"1 RAW TIF 1688x1460",W,H),
                                     cell(s2,"2 CROP TO CHIP BODY",W,H),
                                     cell(s3,"3 STROKE SIGNAL",W,H)},row1);
    cv::hconcat(std::vector<cv::Mat>{cell(s4,"4 BINARY INK (S>0.35)",W,H),
                                     cell(s5,"5 MODEL FREQ MAP",W,H),
                                     cell(s6c,"6 CHAR BOXES (from model)",W,H)},row2);
    cv::Mat row3;
    cv::hconcat(std::vector<cv::Mat>{cell(s7c,"7 ALIGNED BOXES",W,H),
                                     cv::Mat::zeros(H,W,CV_8UC3),
                                     cv::Mat::zeros(H,W,CV_8UC3)},row3);
    cv::Mat out;
    cv::vconcat(std::vector<cv::Mat>{row1,row2,row3},out);
    cv::imwrite(a[3],out);
    std::printf("saved %s\n",a[3].c_str());
    return 0;
}