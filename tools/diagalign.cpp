// 检查：模型字符框 vs 待检实际字形位置的偏差
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<3)return 2;
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;

    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);

        // 整图配准
        cv::Point2d sh(0,0);
        if(!cm.ref.empty()){
            cv::Mat r32,s32;
            cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
            sh=cv::phaseCorrelate(s32,r32,cv::noArray());
            if(std::abs(sh.x)<60&&std::abs(sh.y)<60&&(std::abs(sh.x)>0.01||std::abs(sh.y)>0.01)){
                cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
                cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
            }
        }
        fs::path fp=u8p(a[fi]);
        std::printf("%s  (整图配准 %.1f,%.1f)\n",std::string((const char*)fp.filename().u8string().c_str()).c_str(),-sh.x,-sh.y);

        // 找编号行（区域0：y≈954~1096），检查每个字符框内实际墨量的重心偏移
        auto units=chardet::toUnits(cm,0);
        for(size_t i=0;i<units.size()&&i<3;++i){
            cv::Rect r=units[i].bbox;
            // 扩大搜索区，找实际字形位置
            cv::Rect s(r.x-15,r.y-15,r.width+30,r.height+30);
            s&=cv::Rect(0,0,sig.cols,sig.rows);
            cv::Mat tSig; cv::compare(sig(s),cm.config.strokeHi,tSig,cv::CMP_GT);
            cv::Mat mF; cm.freq(r).convertTo(mF,CV_32F,1.0/255);
            cv::Mat mBin; cv::compare(mF,0.5,mBin,cv::CMP_GT);

            // 模型字形在搜索区中的质心 vs 待检字形质心
            cv::Mat ms=cv::Mat::zeros(s.size(),CV_8U);
            mBin.copyTo(ms(cv::Rect(r.x-s.x,r.y-s.y,r.width,r.height)));
            cv::Moments mm=cv::moments(ms,true), tm=cv::moments(tSig,true);
            if(mm.m00<1||tm.m00<1)continue;
            double mcx=mm.m10/mm.m00, mcy=mm.m01/mm.m00;
            double tcx=tm.m10/tm.m00, tcy=tm.m01/tm.m00;
            std::printf("  #%zu 框(%d,%d) %dx%d  字形偏移 dx=%.1f dy=%.1f  模型墨%d 待检墨%d\n",
                i,r.x,r.y,r.width,r.height,tcx-mcx,tcy-mcy,cv::countNonZero(mBin),cv::countNonZero(tSig));
        }
        std::printf("\n");
    }
    return 0;
}