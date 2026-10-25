// 用精修模型检测产品图：漏字/漏墨/多墨
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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
namespace fs = std::filesystem;
namespace {
std::string w2u(const std::wstring& w){if(w.empty())return{};int n=::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),nullptr,0,nullptr,nullptr);std::string o(n,'\0');::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),o.data(),n,nullptr,nullptr);return o;}
fs::path u8p(const std::string& s){return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
bool rd(const std::string& p, cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read((char*)b.data(),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<3){std::printf("用法: markloc_detect <模型目录> <产品图> [多张...]\n");return 2;}

    chardet::Model cm;
    if(!chardet::loadModel(a[1],cm)){std::printf("模型加载失败: %s\n",a[1].c_str());return 1;}
    std::printf("模型: %d 区域, %d 字符\n\n",(int)cm.regions.size(),cm.totalChars());

    for(size_t fi=2; fi<a.size(); ++fi){
        cv::Mat gray;
        if(!rd(a[fi],gray)){std::printf("%s: 读取失败\n",a[fi].c_str());continue;}
        if(gray.size()!=cm.freq.size()){
            std::printf("%s: 尺寸 %dx%d 与模型 %dx%d 不符\n",a[fi].c_str(),gray.cols,gray.rows,cm.freq.cols,cm.freq.rows);
            continue;
        }

        cv::Mat sig = chardet::strokeSignal(gray,cv::Rect(0,0,gray.cols,gray.rows),cm.config,false);

        // 与 ref 配准
        if(!cm.ref.empty()){
            cv::Mat r32,s32;
            cm.ref.convertTo(r32,CV_32F);
            sig.convertTo(s32,CV_32F);
            cv::Point2d sh = cv::phaseCorrelate(s32,r32,cv::noArray());
            if(std::abs(sh.x)<60&&std::abs(sh.y)<60&&(std::abs(sh.x)>0.01||std::abs(sh.y)>0.01)){
                cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
                cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
            }
        }

        int nMiss=0,nMissInk=0,nExtra=0,nShape=0,total=0;
        double worst=0;
        std::vector<std::string> details;

        std::vector<chardet::CharUnit> allUnits;
        for(size_t ri=0;ri<cm.regions.size();++ri){
            auto units = chardet::toUnits(cm,(int)ri);
            for (const auto& u : units) allUnits.push_back(u);
            auto ms = chardet::match(sig,units,cm.config);
            total += (int)ms.size();
            for(const auto& m : ms){
                if(m.score<=0) continue;
                worst = std::max(worst,m.score);
                if(m.missingChar) ++nMiss;
                else if(m.missScore>=m.score&&m.missScore>0) ++nMissInk;
                else if(m.extraScore>=m.score&&m.extraScore>0) ++nExtra;
                else ++nShape;
                char buf[128];
                std::snprintf(buf,sizeof(buf),"  区域%zu 字符#%d 分%.2f %s",ri+1,m.index+1,m.score,m.label.c_str());
                details.push_back(buf);
            }
        }

        fs::path fp = u8p(a[fi]);
        std::printf("%s\n", fp.filename().u8string().empty()?a[fi].c_str():std::string((const char*)fp.filename().u8string().c_str()).c_str());
        // 整字数量校验
        {
            const auto bd = chardet::diffBlocks(sig, cm.freq, allUnits, cm.config, 8);
            if (bd.valid) {
                std::printf("  整字校验：");
                for (const auto& rc : bd.rows) {
                    std::printf(" [行y%d-%d 本应%d字 实有%d字", rc.y0, rc.y1, rc.expected, rc.found);
                    if (rc.extraChars > 0) std::printf(" 多%d", rc.extraChars);
                    if (rc.missingChars > 0) std::printf(" 缺%d", rc.missingChars);
                    std::printf("]");
                }
                std::printf("\\n");
                std::printf("  字符 %d，最高分 %.2f → %s\\n", total, worst, worst >= 1.0 ? "NG" : "OK");
            } else {
                std::printf("  字符 %d，最高分 %.2f → %s\\n", total, worst, worst >= 1.0 ? "NG" : "OK");
            }
        }
        std::printf("  漏字 %d，漏墨 %d，多墨 %d，字形偏移 %d\n",nMiss,nMissInk,nExtra,nShape);
        std::sort(details.begin(),details.end());
        for(size_t i=0;i<details.size()&&i<5;++i) std::printf("%s\n",details[i].c_str());
        std::printf("\n");
    }
    return 0;
}