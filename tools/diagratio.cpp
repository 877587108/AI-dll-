// 三张异常件的信号强度对比
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
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}

    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        cv::Point2d sh(0,0);
        if(!cm.ref.empty()){cv::Mat r32,s32;cm.ref.convertTo(r32,CV_32F);sig.convertTo(s32,CV_32F);sh=cv::phaseCorrelate(s32,r32,cv::noArray());if(std::abs(sh.x)<60&&std::abs(sh.y)<60&&(std::abs(sh.x)>0.01||std::abs(sh.y)>0.01)){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}}
        fs::path fp=u8p(a[fi]);
        // 编号行所有字符的墨量比
        double sumRatio=0; int cnt=0;
        for(size_t i=0;i<all.size();++i){
            cv::Rect box=all[i].bbox&cv::Rect(0,0,sig.cols,sig.rows);
            if(box.width<=4||box.height<=4)continue;
            constexpr int kS=12;
            cv::Rect search(box.x-kS,box.y-kS,box.width+2*kS,box.height+2*kS);
            search&=cv::Rect(0,0,sig.cols,sig.rows);
            cv::Mat mS,tS; cm.freq(search).convertTo(mS,CV_32F,1.0/255.0); sig(search).convertTo(tS,CV_32F);
            cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-search.x,box.y-search.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
            cv::Point mLoc; double mVal=0; cv::minMaxLoc(res,nullptr,&mVal,nullptr,&mLoc);
            cv::Rect mb(box.x+mLoc.x-kS,box.y+mLoc.y-kS,box.width,box.height);
            if((mb&cv::Rect(0,0,sig.cols,sig.rows))!=mb)continue;
            cv::Mat mb2,tb2;
            cv::compare(cm.freq(box),(int)(cm.config.missFreq*255),mb2,cv::CMP_GT);
            cv::compare(sig(mb),cm.config.strokeHi,tb2,cv::CMP_GT);
            int mi=cv::countNonZero(mb2), ti=cv::countNonZero(tb2);
            if(mi>0){ sumRatio += (double)ti/mi; ++cnt; }
        }
        std::printf("%-40s 配准(%+.1f,%+.1f) 平均墨量比 %.0f%%\n",
            std::string((const char*)fp.filename().u8string().c_str()).substr(0,40).c_str(),
            -sh.x,-sh.y, cnt>0?100.0*sumRatio/cnt:0.0);
    }
    return 0;
}