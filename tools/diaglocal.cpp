// 验证局部缺墨（30x30 小格）能否检出 logo 断裂
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
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){ auto u=chardet::toUnits(cm,(int)ri); for(auto&x:u) all.push_back(x); }
    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        if(!cm.ref.empty()){cv::Mat r32,s32;cm.ref.convertTo(r32,CV_32F);sig.convertTo(s32,CV_32F);auto sh=cv::phaseCorrelate(s32,r32,cv::noArray());if(std::abs(sh.x)<60&&std::abs(sh.y)<60){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}}
        cv::Mat tBin,mBin;
        cv::compare(sig,cm.config.strokeHi,tBin,cv::CMP_GT);
        cv::compare(cm.freq,static_cast<int>(cm.config.missFreq*255),mBin,cv::CMP_GT);
        const int kCell=(fi==2)?45:45;   // 试验：放大到 45
        int found=0; double worst=1e9;
        fs::path fp=u8p(a[fi]);
        std::printf("%s\n",std::string((const char*)fp.filename().u8string().c_str()).c_str());
        for(const auto& u:all){
            cv::Rect b=u.bbox&cv::Rect(0,0,sig.cols,sig.rows);
            if(b.width<kCell*2||b.height<kCell*2)continue;
            for(int y=b.y;y<b.y+b.height;y+=kCell)for(int x=b.x;x<b.x+b.width;x+=kCell){
                cv::Rect cell(x,y,std::min(kCell,b.x+b.width-x),std::min(kCell,b.y+b.height-y));
                if(cell.width<8||cell.height<8)continue;
                int mInk=cv::countNonZero(mBin(cell));
                if(mInk<60)continue;
                int tInk=cv::countNonZero(tBin(cell));
                double ratio=(double)tInk/mInk;
                if(ratio<worst) worst=ratio;
                if(ratio<0.70){
                    ++found;
                    if(found<=12)std::printf("  格(%d,%d) 模型墨%d 待检墨%d 保留%.0f%%\n",cell.x,cell.y,mInk,tInk,ratio*100);
                }
            }
        }
        std::printf("  → 最差保留率 %.0f%% (共%d格<70%%)\n\n", worst<=1e9?worst*100:100.0, found);
    }
    return 0;
}