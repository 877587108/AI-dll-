// 逐字符对齐偏差的完整统计：偏移量、匹配峰值、对齐后墨量比
#include <algorithm>
#include <cstdio>
#include <cmath>
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
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}

    double sumAbsDx=0,sumAbsDy=0,minPeak=1,maxPeak=0; int n=0;
    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        if(!cm.ref.empty()){cv::Mat r32,s32;cm.ref.convertTo(r32,CV_32F);sig.convertTo(s32,CV_32F);auto sh=cv::phaseCorrelate(s32,r32,cv::noArray());if(std::abs(sh.x)<60&&std::abs(sh.y)<60&&(std::abs(sh.x)>0.01||std::abs(sh.y)>0.01)){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}}
        for(auto& u:all){
            cv::Rect box=u.bbox&cv::Rect(0,0,sig.cols,sig.rows);
            if(box.width<=4||box.height<=4)continue;
            constexpr int kS=12;
            cv::Rect search(box.x-kS,box.y-kS,box.width+2*kS,box.height+2*kS);
            search&=cv::Rect(0,0,sig.cols,sig.rows);
            if(search.width<box.width||search.height<box.height)continue;
            cv::Mat mS,tS; cm.freq(search).convertTo(mS,CV_32F,1.0/255.0); sig(search).convertTo(tS,CV_32F);
            cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-search.x,box.y-search.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
            cv::Point mLoc; double mVal=0; cv::minMaxLoc(res,nullptr,&mVal,nullptr,&mLoc);
            int dx=mLoc.x-kS,dy=mLoc.y-kS;
            sumAbsDx+=std::abs(dx); sumAbsDy+=std::abs(dy);
            minPeak=std::min(minPeak,mVal); maxPeak=std::max(maxPeak,mVal);
            ++n;
            if(fi<4 && std::abs(dx)>8) std::printf("  大偏移: dx=%+d dy=%+d 峰值%.2f 框(%d,%d) %dx%d\n",dx,dy,mVal,box.x,box.y,box.width,box.height);
        }
    }
    std::printf("\n统计 %d 个字符:\n",n);
    std::printf("  平均 |dx| = %.2f px, 平均 |dy| = %.2f px\n",sumAbsDx/n,sumAbsDy/n);
    std::printf("  匹配峰值范围 %.2f ~ %.2f\n",minPeak,maxPeak);
    return 0;
}